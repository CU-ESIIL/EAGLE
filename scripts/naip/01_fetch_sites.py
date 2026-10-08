"""Fetch NAIP (all surveys 2011+) and 3DEP-derived rasters for the coregistration test sites onto a common UTM grid.

Usage: python scripts/naip/01_fetch_sites.py CACHE_DIR [site ...]
Writes CACHE_DIR/<site>.npz  (naip_<survey> [4,H,W], lidar layers, grid + metadata json).
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from eagle_als.coreg import data as D  # noqa: E402

SITES = {  # name: (lon, lat, size_m, description)
    "crop_ia": (-93.30, 42.45, 1500, "flat row-crop cropland, central Iowa"),
    "forest_pa": (-79.626, 39.89, 1500, "contiguous closed-canopy hardwood forest, Laurel Highlands PA (hilly)"),
    "mixed_co": (-105.285, 40.00, 1500, "Boulder CO: city, campus, foothills, mixed"),
}


def fetch(name, cache, size=None, suffix=""):
    lon, lat, size0, desc = SITES[name]
    size = size or size0
    out = Path(cache) / f"{name}{suffix}.npz"
    if out.exists():
        print(name, "cached")
        return
    grid = D.Grid.from_center(lon, lat, size, res=1.0)
    t = time.time()
    surveys = D.search_naip(grid)
    arrays, meta = {}, dict(site=name, desc=desc, grid=grid.__dict__, naip={})
    for key, items in surveys.items():
        if int(key[:4]) < 2013:
            continue
        arrays[f"naip_{key}"] = D.read_naip(items, grid)
        meta["naip"][key] = [D.naip_meta(i) for i in items]
        print(name, "naip", key, f"{time.time() - t:.0f}s", flush=True)
    tile = D.tile_for(lon, lat).iloc[0]
    if size > 2000:
        r = D.lidar_rasters_tiled(tile.url, grid, log=lambda m: print(name, m, f"{time.time() - t:.0f}s", flush=True))
        n_pts = -1
    else:
        pts = D.read_lidar_points(tile.url, grid)
        r = D.rasterize_lidar(pts, grid)
        n_pts = len(pts)
    meta["lidar"] = dict(project=tile.name, year=int(tile.collection_year), n_points=int(n_pts), ql=tile.ql)
    print(name, "lidar", meta["lidar"], f"{time.time() - t:.0f}s", flush=True)
    for k in ("dsm", "dtm", "chm", "intensity", "density"):
        arrays[k] = getattr(r, k)
    np.savez_compressed(out, meta=json.dumps(meta), **arrays)


if __name__ == "__main__":
    cache = sys.argv[1]
    Path(cache).mkdir(parents=True, exist_ok=True)
    big = [a for a in sys.argv[2:] if a.startswith("--size=")]  # e.g. --size=5000 writes <site>_5000.npz
    names = [a for a in sys.argv[2:] if not a.startswith("--")]
    for n in names or SITES:
        if big:
            fetch(n, cache, int(big[0].split("=")[1]), "_" + big[0].split("=")[1])
        else:
            fetch(n, cache)
