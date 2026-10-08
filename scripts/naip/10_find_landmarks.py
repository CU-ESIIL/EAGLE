"""Find verified building landmarks in a 5 km site: CHM candidates -> 0.5 m lidar chip check -> pickle.

Usage: python scripts/naip/10_find_landmarks.py CACHE_DIR STEM OUT_PKL [MAX_CANDIDATES]
Each landmark: row/col (1 m site grid), x/y (UTM), chip grid, roof mask (0.5 m), roof height, area, orientation.
"""

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from eagle_als.coreg import data as D  # noqa: E402
from eagle_als.coreg import landmarks as L  # noqa: E402

if __name__ == "__main__":
    cache, stem, out = sys.argv[1:4]
    nmax = int(sys.argv[4]) if len(sys.argv) > 4 else 10**9
    d = np.load(Path(cache) / f"{stem}.npz")
    meta = json.loads(str(d["meta"]))
    grid = D.Grid(**meta["grid"])
    cand = L.candidates(d["chm"])
    rng = np.random.default_rng(0)
    if len(cand) > nmax:
        cand = [cand[i] for i in rng.choice(len(cand), nmax, replace=False)]
    from pyproj import Transformer

    lonlat = Transformer.from_crs(grid.crs, "EPSG:4269", always_xy=True)
    tile = D.tile_for(*lonlat.transform(grid.x0 + grid.width / 2, grid.y0 - grid.height / 2)).iloc[0]
    found, t0 = [], time.time()
    for i, (r, c, a, h) in enumerate(cand):
        x, y = grid.x0 + (c + 0.5) * grid.res, grid.y0 - (r + 0.5) * grid.res
        chip = D.Grid(grid.crs, x - 40, y + 40, 0.5, 160, 160)
        try:
            pts = D.read_lidar_points(tile.url, chip, pad=1)
            v = L.verify_chip(pts, chip)
        except Exception as e:  # noqa: BLE001
            print("fail", i, e)
            continue
        if v:
            found.append(dict(row=r, col=c, x=x, y=y, chip=chip.__dict__, **v))
        if i % 25 == 0:
            print(f"{stem}: {i}/{len(cand)} candidates, {len(found)} landmarks, {time.time() - t0:.0f}s", flush=True)
    pickle.dump(dict(landmarks=found, tile=tile.name), open(out, "wb"))
    print(f"{stem}: {len(found)} verified landmarks of {len(cand)} candidates")
