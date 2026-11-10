"""Coordinate reference systems for dataset ingestion: read it, never guess it; output WGS84 lat/lon.

Every evaluation table stores `lat`, `lon` as WGS84 decimal degrees (EPSG:4326), because that is what the 3DEP
footprint lookup and the cookie cache take. Raw data come in whatever the source used, so each ingestion does:

1. Find the source CRS in the data itself (`source_crs`: file metadata, .prj, GeoPackage/Parquet CRS) or in the
   dataset's documentation, and write it down in the dataset README. Plain lat/lon columns have no CRS in the
   file; the documentation must say which datum. `to_wgs84` has no default, so a missing CRS is an error.
2. Convert with `to_wgs84(x, y, src_crs)` (x = easting or lon, y = northing or lat).
3. Run `check_lonlat(lat, lon)` and write its counts to `catalog/crs_check.csv`: it flags out-of-range values,
   swapped lat/lon, a lost minus sign on western longitudes, (0, 0) and values outside the expected region.
4. If the CRS is missing or doubtful, `candidate_crs(x, y, expected_bbox)` lists which candidate CRSs put the
   points inside the region the dataset covers. That is evidence to confirm with the data provider, not a decision.

Datum note: NAD83 (EPSG:4269) to WGS84 differs by about 1 m over CONUS and pyproj applies no shift without grid
files, so NAD83 lat/lon are accepted as WGS84 (far below the 100 m cookie radius). Say so in the README.
"""

import numpy as np
import pandas as pd

OUTPUT_CRS = "EPSG:4326"
CONUS_BBOX = (-125.0, 24.0, -66.5, 50.0)  # lon_min, lat_min, lon_max, lat_max
US_BBOX = [(-180.0, 17.0, -64.0, 72.0),  # CONUS, Alaska, Hawaii, Puerto Rico
           (144.0, 13.0, 146.5, 21.0)]  # Guam and the Marianas


def source_crs(path, layer=None):
    """CRS stored in a vector file (.shp/.prj, .gpkg, .gdb, GeoParquet). Raises if the file has none."""
    import pyogrio

    crs = pyogrio.read_info(path, layer=layer).get("crs")
    if not crs:
        raise ValueError(f"{path} has no CRS recorded; take it from the dataset documentation and pass it explicitly")
    return crs


def to_wgs84(x, y, src_crs):
    """Transform x, y (easting/northing or lon/lat) from `src_crs` to WGS84. Returns (lon, lat) arrays.

    `src_crs` is required (EPSG code, string or pyproj CRS); there is no default on purpose.
    """
    from pyproj import CRS, Transformer

    if src_crs is None:
        raise ValueError("src_crs is required: read it from the data or its documentation, do not assume it")
    src = CRS.from_user_input(src_crs)
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    lon, lat = Transformer.from_crs(src, OUTPUT_CRS, always_xy=True).transform(x, y)
    lon, lat = np.asarray(lon, dtype=float), np.asarray(lat, dtype=float)
    bad = ~np.isfinite(lon) | ~np.isfinite(lat)  # pyproj returns inf for failures
    lon[bad], lat[bad] = np.nan, np.nan
    return lon, lat


def check_lonlat(lat, lon, bbox=US_BBOX):
    """Count suspicious coordinates. Returns a DataFrame (check, n_rows) for catalog/crs_check.csv.

    bbox = (lon_min, lat_min, lon_max, lat_max), or a list of them, covering where the dataset should be.
    """
    lat, lon = pd.Series(np.asarray(lat, dtype=float)), pd.Series(np.asarray(lon, dtype=float))
    boxes = [bbox] if isinstance(bbox[0], (int, float)) else list(bbox)

    def within(la, lo):
        m = pd.Series(False, index=la.index)
        for x0, y0, x1, y1 in boxes:
            m |= la.between(y0, y1) & lo.between(x0, x1)
        return m

    has = lat.notna() & lon.notna()
    in_range = lat.between(-90, 90) & lon.between(-180, 180)
    inside = within(lat, lon)
    swapped = has & ~inside & within(lon, lat)
    sign = has & ~inside & within(lat, -lon)
    rows = [
        ("rows", len(lat)),
        ("missing lat or lon", int((~has).sum())),
        ("outside +/-90, +/-180", int((has & ~in_range).sum())),
        ("(0, 0)", int((has & (lat == 0) & (lon == 0)).sum())),
        ("outside expected region", int((has & in_range & ~inside).sum())),
        ("  of which look like swapped lat/lon", int(swapped.sum())),
        ("  of which look like a lost minus sign on lon", int(sign.sum())),
    ]
    return pd.DataFrame(rows, columns=["check", "n_rows"])


def candidate_crs(x, y, bbox=CONUS_BBOX, candidates=None, sample=1000):
    """Which candidate CRSs put the points inside `bbox` (lon/lat)? Returns DataFrame (crs, frac_inside).

    For data with no stated CRS only. Several can fit (e.g. NAD83 vs WGS84 UTM); the documentation decides.
    """
    from pyproj import CRS

    if candidates is None:
        candidates = ["EPSG:4326", "EPSG:5070", "EPSG:3310", "EPSG:3857", "EPSG:2163"] + [
            f"EPSG:{base + z}" for base in (26900, 32600) for z in range(1, 20)
        ]
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    idx = np.flatnonzero(ok)[:sample]
    x0, y0, x1, y1 = bbox
    rows = []
    for c in candidates:
        lon, lat = to_wgs84(x[idx], y[idx], CRS.from_user_input(c))
        inside = np.isfinite(lon) & (lon >= x0) & (lon <= x1) & (lat >= y0) & (lat <= y1)
        rows.append((c, float(inside.mean()) if len(idx) else 0.0))
    return pd.DataFrame(rows, columns=["crs", "frac_inside"]).sort_values("frac_inside", ascending=False)
