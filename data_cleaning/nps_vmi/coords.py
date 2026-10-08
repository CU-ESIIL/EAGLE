"""Shared coordinate repair for VMI plots: raw X/Y (+ zone, datum) -> lon/lat with QA flags.

Repairs (each recorded in flags): lat/lon given as (lat, lon) with lost lon sign; northing in the X
column; wrong or missing UTM zone (resolved by picking the zone that lands inside the park/project
bounding box from the IRMA profile). Nothing is dropped; unresolvable rows get NaN lon/lat.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely import wkt

RAW = Path("datasets/raw/nps_vmi")
_bbox = {}


def bbox_for(package):
    if package not in _bbox:
        b = []
        f = RAW / "api_cache" / f"profile_{package.split('_')[1]}.json"
        if f.exists():
            for w in json.load(open(f)).get("boundingBoxes") or []:
                if w.get("wkt"):
                    b.append(wkt.loads(w["wkt"]).bounds)
        _bbox[package] = (min(v[0] for v in b), min(v[1] for v in b), max(v[2] for v in b), max(v[3] for v in b)) if b else None
    return _bbox[package]


def _dist(b, lon, lat):
    if b is None:
        return np.nan
    return float(np.hypot(max(b[0] - lon, 0, lon - b[2]), max(b[1] - lat, 0, lat - b[3])))


def _utm(x, y, z, nad83):
    return Transformer.from_crs((26900 if nad83 else 32600) + int(z), 4326, always_xy=True).transform(x, y)


def to_lonlat(package, x, y, zone, nad83):
    """Arrays/Series for one package -> (lon, lat, flags list, bbox_dist_deg array)."""
    x, y, zone = (np.asarray(pd.to_numeric(a, errors="coerce"), dtype=float) for a in (x, y, zone))
    zone = np.where((zone >= 1) & (zone <= 60), zone, np.nan)  # zone 0 / junk = missing
    nad83 = np.broadcast_to(np.asarray(nad83, dtype=bool), x.shape)
    b = bbox_for(package)
    n = len(x)
    lon, lat, dist = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    flags = [""] * n
    swap = (x > 1_000_000) & (y < 1_000_000)  # northing recorded in the X column
    x, y = np.where(swap, y, x), np.where(swap, x, y)
    is_ll = (np.abs(x) <= 180) & (np.abs(y) <= 90)
    for i in range(n):
        if np.isnan(x[i]) or np.isnan(y[i]):
            flags[i] = "no_coordinates;"
            continue
        if swap[i]:
            flags[i] += "utm_xy_swapped;"
        if is_ll[i]:
            # candidate (lon, lat, flag): as given; lon sign lost; (lat, lon) order with/without sign
            cands = [(x[i], y[i], "")]
            if abs(x[i]) >= 17 and abs(y[i]) <= 72 and x[i] > 0:
                cands.append((-x[i], y[i], "lon_sign_assumed;"))
            if abs(x[i]) < 72 and abs(y[i]) > 60:
                cands.append((-abs(y[i]), abs(x[i]), "latlon_swapped_lon_sign_assumed;"))
            if b:
                dists = [_dist(b, c[0], c[1]) for c in cands]
                lon[i], lat[i], fl = cands[int(np.argmin(dists))]
            else:  # no bbox: prefer the first candidate that is plausible for the US
                lon[i], lat[i], fl = next((c for c in cands if 17 <= c[1] <= 72 and -180 <= c[0] <= -60), cands[0])
            flags[i] += fl
        else:
            best = None
            if not np.isnan(zone[i]):
                lo, la = _utm(x[i], y[i], zone[i], nad83[i])
                best = (_dist(b, lo, la), zone[i], lo, la)
            if (best is None or best[1 - 1] > 0.25) and b:  # wrong or missing zone label
                for z in range(int((b[0] + 180) // 6), int((b[2] + 180) // 6) + 3):
                    if 1 <= z <= 60 and (best is None or z != best[1]):
                        lo, la = _utm(x[i], y[i], z, nad83[i])
                        d = _dist(b, lo, la)
                        if best is None or d < best[0] - 1e-9:
                            flags[i] += "utm_zone_inferred;" if np.isnan(zone[i]) else "utm_zone_corrected;"
                            best = (d, z, lo, la)
            if best is None:
                flags[i] += "utm_zone_missing;"
                continue
            lon[i], lat[i] = best[2], best[3]
        dist[i] = _dist(b, lon[i], lat[i])
        if dist[i] > 0.25:
            flags[i] += "outside_park_bbox;"
    flags = [";".join(dict.fromkeys(f for f in t.split(";") if f)) for t in flags]
    return lon, lat, flags, dist
