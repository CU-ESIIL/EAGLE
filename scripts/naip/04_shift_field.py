"""Compute block x height-class regression surfaces over a large area (parallel) and save them for field analysis.

For each NAIP vintage: fit sun on a central window, pooled initial shift on sampled blocks, then `class_surfaces` over the
whole grid with `--jobs` processes. Output: CACHE_DIR/surfaces_<stem>_<vintage>.pkl (dict: sun, init, surfaces).
Analysis of those surfaces (windowed fits, polynomial fields) is in 05_field_analysis.py.

Usage: python scripts/naip/04_shift_field.py CACHE_DIR SITE_STEM JOBS VINTAGE [VINTAGE ...]   e.g. forest_pa_5000 8 2019_0.6m
"""

import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from eagle_als.coreg import api  # noqa: E402
from eagle_als.coreg import model as M  # noqa: E402
from eagle_als.coreg import register as R  # noqa: E402
from eagle_als.coreg.data import LidarRasters  # noqa: E402

if __name__ == "__main__":
    cache, stem, jobs = sys.argv[1], sys.argv[2], int(sys.argv[3])
    d = np.load(Path(cache) / f"{stem}.npz")
    lr = LidarRasters(d["dsm"], d["dtm"], d["chm"], d["intensity"], d["density"])
    H, W = lr.chm.shape
    rng = np.random.default_rng(0)
    for v in sys.argv[4:]:
        t0 = time.time()
        naip = d[f"naip_{v}"]
        Y, X = api.build(naip, lr)
        sun = None
        if api.tall_fraction(lr.chm) > 0.05:
            sl = (slice(H // 2 - 400, H // 2 + 400), slice(W // 2 - 400, W // 2 + 400))
            az, el, _ = M.fit_sun(Y[:, sl[0], sl[1]], lr.dsm[sl], X[:, sl[0], sl[1]], radius=8, window=500)
            sun = (az, el)
            Y, X = api.build(naip, lr, sun)
        blocks = R.block_estimates(Y, X, 320, 12)
        blocks = [blocks[i] for i in rng.choice(len(blocks), min(25, len(blocks)), replace=False)]
        init, _ = R.pool_surfaces(blocks, 12)
        print(f"{v}: sun {sun} init ({init.dx:+.2f},{init.dy:+.2f}) {time.time() - t0:.0f}s", flush=True)
        surfs = M.class_surfaces(Y, X, lr.chm, (init.dx, init.dy), block=250, radius=10, n_jobs=jobs)
        pickle.dump(dict(sun=sun, init=(init.dx, init.dy), surfaces=surfs, shape=(H, W)), open(Path(cache) / f"surfaces_{stem}_{v}.pkl", "wb"))
        print(f"{v}: {len(surfs)} surfaces {time.time() - t0:.0f}s", flush=True)
