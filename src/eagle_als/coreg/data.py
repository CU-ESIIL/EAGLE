"""Data access for NAIP <-> 3DEP coregistration.

Everything is returned on one analysis grid (UTM, NAD83, integer-metre-aligned, `res` m pixels) so that
no resampling of either product is needed afterwards and shifts can be reported directly in metres.

NAIP:  read from Planetary Computer COGs, area-averaged onto the analysis grid (no geometric change).
3DEP:  streamed from the USGS EPT with PDAL; rasterised to DSM / DTM / CHM / first-return intensity / etc.

Datum note: USGS EPT stores X/Y as EPSG:3857 with `+nadgrids=@null`, i.e. the lon/lat inside it are the
original NAD83 values labelled as WGS84. We therefore undo the Mercator and apply UTM on the NAD83 datum
with *no* datum shift, so the lidar and NAIP (NAD83 UTM) share a datum.
"""

import json
import time
from dataclasses import dataclass, field

import numpy as np

EPT_PROJ = "+proj=merc +a=6378137 +b=6378137 +lat_ts=0 +lon_0=0 +x_0=0 +y_0=0 +k=1 +units=m +nadgrids=@null +no_defs"
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
LIDAR_DIMS = ["X", "Y", "Z", "Intensity", "ReturnNumber", "NumberOfReturns", "Classification", "PointSourceId", "ScanAngleRank"]


@dataclass
class Grid:
    """Analysis grid: north-up, `res` m pixels, upper-left corner (x0, y0) in `crs`."""

    crs: str
    x0: float
    y0: float
    res: float
    width: int
    height: int

    @property
    def transform(self):
        from rasterio.transform import Affine

        return Affine(self.res, 0, self.x0, 0, -self.res, self.y0)

    @property
    def bounds(self):
        return (self.x0, self.y0 - self.height * self.res, self.x0 + self.width * self.res, self.y0)

    @classmethod
    def from_center(cls, lon, lat, size_m, res=1.0):
        """Square grid of side ~size_m centred on lon/lat, in the NAD83 UTM zone of that point."""
        from pyproj import Transformer

        zone = int((lon + 180) // 6) + 1
        crs = f"EPSG:{26900 + zone}"
        x, y = Transformer.from_crs("EPSG:4269", crs, always_xy=True).transform(lon, lat)
        n = int(round(size_m / res))
        x0 = np.floor((x - size_m / 2) / res) * res
        y0 = np.ceil((y + size_m / 2) / res) * res
        return cls(crs, float(x0), float(y0), float(res), n, n)


def search_naip(grid, years=None):
    """NAIP surveys intersecting the grid as {survey_key: [signed STAC items]}, newest first.

    A survey is all scenes (quarter-quads) sharing a collection date/year; a grid on a scene boundary needs several.
    """
    import planetary_computer
    import pystac_client
    from pyproj import Transformer

    x0, y0, x1, y1 = grid.bounds
    west, south, east, north = Transformer.from_crs(grid.crs, "EPSG:4326", always_xy=True).transform_bounds(x0, y0, x1, y1)
    cat = pystac_client.Client.open(STAC_URL)  # unsigned: hrefs are signed at read time (tokens expire after ~1 h)
    items = list(cat.search(collections=["naip"], bbox=[west, south, east, north]).items())
    if years is not None:
        items = [i for i in items if int(i.properties["naip:year"]) in set(years)]
    surveys = {}
    for i in sorted(items, key=lambda i: i.datetime, reverse=True):
        key = f"{i.properties['naip:year']}_{i.properties.get('gsd')}m"
        surveys.setdefault(key, []).append(i)
    return surveys


def read_naip(items, grid):
    """Read NAIP RGBN on `grid` (area-averaged). `items` is one STAC item or a list (mosaicked, first wins).

    Returns float32 [4, H, W] with NaN where there is no data. NAIP is natively NAD83 UTM, so when the scene is in the
    grid's CRS this is pure block-averaging (no geometric change); other zones are bilinearly reprojected.
    """
    import planetary_computer
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    items = [items] if not isinstance(items, (list, tuple)) else items
    out = np.full((4, grid.height, grid.width), np.nan, np.float32)
    for item in items:
        with rasterio.open(planetary_computer.sign(item.assets["image"].href)) as src:
            rs = Resampling.average if src.crs.to_string() == grid.crs else Resampling.bilinear
            for b in range(4):
                if np.isfinite(out[b]).all():
                    continue
                dst = np.full((grid.height, grid.width), np.nan, np.float32)
                for attempt in range(4):  # remote COG reads occasionally fail transiently
                    try:
                        reproject(rasterio.band(src, b + 1), dst, dst_transform=grid.transform, dst_crs=grid.crs,
                                  src_nodata=0, dst_nodata=np.nan, resampling=rs)
                        break
                    except rasterio.errors.RasterioError:
                        if attempt == 3:
                            raise
                        time.sleep(2 * (attempt + 1))
                out[b] = np.where(np.isfinite(out[b]), out[b], dst)
    return out


def naip_meta(item):
    p = item.properties
    return dict(id=item.id, date=str(item.datetime.date()), year=int(p["naip:year"]), gsd=p.get("gsd"), crs=p.get("proj:code"))


def tile_for(lon, lat):
    """Name of the (newest) 3DEP EPT project containing lon/lat."""
    from eagle_als.fetch import tile_index
    from shapely.geometry import Point

    t = tile_index()
    t = t[t.geometry.contains(Point(lon, lat)) & t.collection_year.notna()]
    if not len(t):
        raise ValueError("no 3DEP project here")
    return t.sort_values("collection_year", ascending=False)


def read_lidar_points(tile_url, grid, pad=20.0, resolution=None, timeout=None):
    """Stream points inside the grid (+pad m) as a structured array in grid.crs (NAD83 UTM). Noise classes dropped."""
    import pdal
    from pyproj import Transformer

    x0, y0, x1, y1 = grid.bounds
    x0, y0, x1, y1 = x0 - pad, y0 - pad, x1 + pad, y1 + pad
    tr = Transformer.from_crs(grid.crs, EPT_PROJ, always_xy=True)  # prefilter box in EPT Mercator (pad for scale factor)
    xs, ys = tr.transform([x0, x1, x0, x1], [y0, y0, y1, y1])
    cx, cy = np.mean(xs), np.mean(ys)
    half = max(np.ptp(xs), np.ptp(ys)) / 2 * 1.05
    reader = {
        "type": "readers.ept",
        "filename": tile_url,
        "bounds": f"([{cx - half}, {cx + half}], [{cy - half}, {cy + half}])",
    }
    if resolution is not None:
        reader["resolution"] = resolution
    if timeout is not None:
        reader["timeout"] = int(timeout)
    stages = [
        reader,
        {"type": "filters.range", "limits": "Classification![7:7],Classification![18:18]"},
        {"type": "filters.reprojection", "in_srs": EPT_PROJ, "out_srs": grid.crs},
        {"type": "filters.crop", "bounds": f"([{x0}, {x1}], [{y0}, {y1}])"},
    ]
    for attempt in range(4):  # S3 reads occasionally fail transiently
        try:
            pipe = pdal.Pipeline(json.dumps({"pipeline": stages}))
            pipe.execute(allowed_dims=LIDAR_DIMS)
            return pipe.arrays[0]
        except RuntimeError:
            if attempt == 3:
                raise
            time.sleep(5 * (attempt + 1))


@dataclass
class LidarRasters:
    """Lidar-derived layers on the analysis grid (all float32 [H, W], NaN where empty)."""

    dsm: np.ndarray  # max Z of all returns
    dtm: np.ndarray  # mean Z of class-2 ground returns, gap filled
    chm: np.ndarray  # dsm - dtm
    intensity: np.ndarray  # mean intensity of first returns (1064 nm), normalised per project (median=1)
    density: np.ndarray  # returns per m2
    layers: dict = field(default_factory=dict)


def rasterize_lidar(pts, grid):
    from scipy import ndimage

    H, W, res = grid.height, grid.width, grid.res
    col = np.floor((pts["X"] - grid.x0) / res).astype(np.int64)
    row = np.floor((grid.y0 - pts["Y"]) / res).astype(np.int64)
    ok = (col >= 0) & (col < W) & (row >= 0) & (row < H)
    pts, col, row = pts[ok], col[ok], row[ok]
    flat = row * W + col
    n = H * W
    z = pts["Z"]

    # DSM: max Z per cell
    dsm = np.full(n, -np.inf)
    np.maximum.at(dsm, flat, z)
    dsm[~np.isfinite(dsm)] = np.nan

    # DTM: mean Z of ground points, then fill from nearest ground cell and smooth
    g = pts["Classification"] == 2
    gc = np.bincount(flat[g], minlength=n)
    gz = np.bincount(flat[g], weights=z[g], minlength=n)
    dtm = np.where(gc > 0, gz / np.maximum(gc, 1), np.nan).reshape(H, W)
    if np.isfinite(dtm).any():
        idx = ndimage.distance_transform_edt(~np.isfinite(dtm), return_distances=False, return_indices=True)
        dtm = ndimage.gaussian_filter(dtm[tuple(idx)], 1.5)
    dsm = dsm.reshape(H, W)
    chm = np.clip(dsm - dtm, 0, None)

    # first-return intensity (mean)
    fr = pts["ReturnNumber"] == 1
    ic = np.bincount(flat[fr], minlength=n)
    isum = np.bincount(flat[fr], weights=pts["Intensity"][fr].astype(np.float64), minlength=n)
    inten = np.where(ic > 0, isum / np.maximum(ic, 1), np.nan).reshape(H, W)
    med = np.nanmedian(inten)
    inten = inten / med if med > 0 else inten
    density = (np.bincount(flat, minlength=n) / res**2).reshape(H, W)
    return LidarRasters(
        dsm.astype(np.float32), dtm.astype(np.float32), chm.astype(np.float32), inten.astype(np.float32), density.astype(np.float32)
    )


def lidar_rasters_tiled(tile_url, grid, tile=1000, pad=10.0, log=print):
    """Rasterise a large grid by streaming it in `tile` m chunks (bounded memory). Returns LidarRasters on `grid`.

    Intensity is normalised per chunk to the chunk median, so project-wide intensity trends are removed (fine for matching).
    """
    H, W = grid.height, grid.width
    names = ("dsm", "dtm", "chm", "intensity", "density")
    out = {n: np.full((H, W), np.nan, np.float32) for n in names}
    step = int(round(tile / grid.res))
    for r0 in range(0, H, step):
        for c0 in range(0, W, step):
            h, w = min(step, H - r0), min(step, W - c0)
            sub = Grid(grid.crs, grid.x0 + c0 * grid.res, grid.y0 - r0 * grid.res, grid.res, w, h)
            pts = read_lidar_points(tile_url, sub, pad=pad)
            if len(pts) == 0:
                continue
            r = rasterize_lidar(pts, sub)
            for n in names:
                out[n][r0 : r0 + h, c0 : c0 + w] = getattr(r, n)
            log(f"lidar chunk r{r0} c{c0}: {len(pts)/1e6:.1f}M pts")
    return LidarRasters(**out)
