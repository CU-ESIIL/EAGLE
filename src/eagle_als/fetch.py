"""Fetch cylindrical ALS "cookies" from USGS 3DEP EPT tiles with PDAL and store them locally.

Based on `scripts/3dep/dataset.py::load_als_cookie`. Each cookie is reprojected to local UTM,
centered on the query location (x=y=0), and saved as an uncompressed `.npz` with:

    xyz                float32 [N, 3]  x, y relative to the query point (m), z = elevation (m)
    hag                float32 [N]     height above ground (m), from ground-classified points
    intensity          uint16  [N]
    return_number      uint8   [N]
    number_of_returns  uint8   [N]
    classification     uint8   [N]
    meta               json string (tile, lat, lon, epsg, radius, n_points, has_ground, ...)
"""

import json
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
TILE_INDEX = REPO_ROOT / "datasets/USGS_3dep/usgs_3dep_resources_AWS.geojson"
EPT_STORAGE_CRS = "EPSG:3857"  # USGS public EPT tiles are indexed/stored in Web Mercator
DIMS = ["X", "Y", "Z", "Intensity", "ReturnNumber", "NumberOfReturns", "Classification"]
GROUND_CLASS = 2
DROP_CLASSES = (7, 18)  # low noise, high noise

_TILES = None


def tile_index():
    """Lazily loaded 3DEP tile index (GeoDataFrame indexed by tile name)."""
    global _TILES
    if _TILES is None:
        import geopandas as gpd

        _TILES = gpd.read_file(TILE_INDEX).set_index("name")
    return _TILES


def utm_epsg(lon, lat):
    """EPSG code of the WGS84 UTM zone containing lon/lat."""
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def fetch_cookie(tile_name, lat, lon, radius=100.0, timeout=None):
    """Stream a cylinder of `radius` meters around lat/lon from a 3DEP EPT tile.

    Returns a dict of numpy arrays (see module docstring); raises on PDAL/network errors.
    """
    import pdal
    from pyproj import Transformer
    from shapely.geometry import Point

    tile = tile_index().loc[tile_name]
    if not tile.geometry.contains(Point(lon, lat)):
        raise ValueError("Query point is outside the tile's footprint.")

    epsg = utm_epsg(lon, lat)
    # coarse prefilter in the EPT storage CRS (distorted distances -> pad the radius)
    sx, sy = Transformer.from_crs("EPSG:4326", EPT_STORAGE_CRS, always_xy=True).transform(lon, lat)
    # Web Mercator scale factor is 1/cos(lat); pad generously so the exact UTM crop is covered
    pad = radius * 1.2 / np.cos(np.radians(lat))
    bounds = f"([{sx - pad}, {sx + pad}], [{sy - pad}, {sy + pad}])"
    ux, uy = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True).transform(lon, lat)

    stages = [
        {"type": "readers.ept", "filename": tile.url, "bounds": bounds},
        {"type": "filters.range", "limits": ",".join(f"Classification![{c}:{c}]" for c in DROP_CLASSES)},
        {"type": "filters.reprojection", "in_srs": EPT_STORAGE_CRS, "out_srs": f"EPSG:{epsg}"},
        {"type": "filters.crop", "point": f"POINT({ux} {uy})", "distance": radius},
    ]
    if timeout is not None:
        stages[0]["timeout"] = int(timeout)
    pipeline = pdal.Pipeline(json.dumps({"pipeline": stages}))
    pipeline.execute(allowed_dims=DIMS)
    arrays = pipeline.arrays
    if len(arrays) == 0 or len(arrays[0]) == 0:
        raise EmptyCookieError("No points returned for this location.")
    pts = arrays[0]

    xyz = np.stack([pts["X"] - ux, pts["Y"] - uy, pts["Z"]], axis=1).astype(np.float32)
    cls = pts["Classification"].astype(np.uint8)
    hag, has_ground = height_above_ground(xyz, cls)
    cookie = dict(
        xyz=xyz,
        hag=hag,
        intensity=pts["Intensity"].astype(np.uint16),
        return_number=pts["ReturnNumber"].astype(np.uint8),
        number_of_returns=pts["NumberOfReturns"].astype(np.uint8),
        classification=cls,
    )
    meta = dict(
        tile=tile_name,
        lat=float(lat),
        lon=float(lon),
        epsg=epsg,
        radius=float(radius),
        n_points=int(len(xyz)),
        has_ground=bool(has_ground),
        collection_year=None if np.isnan(tile.collection_year) else int(tile.collection_year),
        ql=None if not isinstance(tile.ql, str) else tile.ql,
    )
    return cookie, meta


class EmptyCookieError(RuntimeError):
    pass


def height_above_ground(xyz, classification, k=8, cell=2.0):
    """Height above ground by inverse-distance interpolation of ground (class 2) points.

    Falls back to a coarse "lowest point per cell" ground surface when the tile has no
    ground classification. Returns (hag float32 [N], has_ground bool).
    """
    from scipy.spatial import cKDTree

    ground = classification == GROUND_CLASS
    has_ground = ground.sum() >= 10
    if has_ground:
        gxyz = xyz[ground]
    else:
        # lowest point in each `cell` m column as a pseudo ground surface
        ij = np.floor(xyz[:, :2] / cell).astype(np.int64)
        key = ij[:, 0] * 100003 + ij[:, 1]
        order = np.lexsort((xyz[:, 2], key))
        first = np.ones(len(order), dtype=bool)
        first[1:] = key[order][1:] != key[order][:-1]
        gxyz = xyz[order[first]]
    tree = cKDTree(gxyz[:, :2])
    k = min(k, len(gxyz))
    dist, idx = tree.query(xyz[:, :2], k=k)
    if k == 1:
        dist, idx = dist[:, None], idx[:, None]
    w = 1.0 / np.maximum(dist, 0.1)
    ground_z = (gxyz[idx, 2] * w).sum(1) / w.sum(1)
    hag = (xyz[:, 2] - ground_z).astype(np.float32)
    return hag, has_ground


def save_cookie(path, cookie, meta):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, meta=np.array(json.dumps(meta)), **cookie)
    tmp.rename(path)  # atomic: readers never see partial files


def load_cookie(path):
    with np.load(path) as f:
        cookie = {k: f[k] for k in f.files if k != "meta"}
        meta = json.loads(str(f["meta"]))
    return cookie, meta


def thin_cookie(cookie, meta, max_points, seed=0):
    """Randomly subsample a cookie to at most `max_points` (keeps very dense tiles manageable)."""
    n = len(cookie["xyz"])
    if max_points is None or n <= max_points:
        return cookie, meta
    keep = np.sort(np.random.default_rng(seed).choice(n, max_points, replace=False))
    meta = dict(meta, n_points=int(max_points), n_points_raw=int(n))
    return {k: v[keep] for k, v in cookie.items()}, meta


def fetch_and_save(path, tile_name, lat, lon, radius=100.0, min_points=100, max_points=None, timeout=None):
    """Fetch one cookie and save it. Returns a status record (never raises)."""
    t0 = time.time()
    rec = dict(path=str(path), tile=tile_name, lat=lat, lon=lon, status="ok", n_points=0, error="")
    try:
        cookie, meta = fetch_cookie(tile_name, lat, lon, radius=radius, timeout=timeout)
        rec["n_points"] = meta["n_points"]
        rec["has_ground"] = meta["has_ground"]
        if meta["n_points"] < min_points:
            rec["status"] = "too_few_points"
        else:
            validate_cookie(cookie)
            cookie, meta = thin_cookie(cookie, meta, max_points)
            save_cookie(path, cookie, meta)
    except EmptyCookieError as e:
        rec["status"], rec["error"] = "empty", str(e)
    except KeyError as e:
        rec["status"], rec["error"] = "unknown_tile", repr(e)
    except Exception as e:  # network / PDAL / corrupted tile errors
        rec["status"], rec["error"] = "error", f"{type(e).__name__}: {e}"[:500]
    rec["seconds"] = round(time.time() - t0, 2)
    return rec


def validate_cookie(cookie):
    """Raise ValueError for obviously corrupted point clouds."""
    xyz = cookie["xyz"]
    if not np.isfinite(xyz).all():
        raise ValueError("non-finite coordinates")
    z = xyz[:, 2]
    if z.max() - z.min() > 3000:
        raise ValueError(f"implausible elevation range {z.max() - z.min():.0f} m")
    if not np.isfinite(cookie["hag"]).all():
        raise ValueError("non-finite height above ground")
