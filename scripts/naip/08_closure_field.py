"""Vintage closure for the large-area field models.

For every pair of NAIP vintages, the displacement between them is measured directly in 500 m windows (NAIP vs NAIP, hp
luminance NCC, windows with NCC > 0.3), and compared with what the lidar-derived field models predict for the same
windows: the mean over the window of (D_a - D_b), where D = ground field + lidar height x lean field, weighted by the
luminance texture energy that drives the NCC. Residual RMS (m) is the closure error; "naive" assumes D = 0 for both.

Usage: python scripts/naip/08_closure_field.py CACHE_DIR STEM VINTAGE [VINTAGE ...] (needs surfaces_<STEM>_<VINTAGE>.pkl)
"""

import itertools
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from eagle_als.coreg import api  # noqa: E402
from eagle_als.coreg import features as F  # noqa: E402
from eagle_als.coreg import model as M  # noqa: E402
from eagle_als.coreg import register as R  # noqa: E402
from eagle_als.coreg.data import LidarRasters  # noqa: E402

WIN, RAD = 500, 14
MEASURE = "hp"  # set by env CLOSURE_MEASURE=edges
import os

MEASURE = os.environ.get("CLOSURE_MEASURE", MEASURE)

if __name__ == "__main__":
    cache, stem, vints = sys.argv[1], sys.argv[2], sys.argv[3:]
    d = np.load(Path(cache) / f"{stem}.npz")
    lr = LidarRasters(d["dsm"], d["dtm"], d["chm"], d["intensity"], d["density"])
    H, W = lr.chm.shape
    fields, lum = {}, {}
    for v in vints:
        p = pickle.load(open(Path(cache) / f"surfaces_{stem}_{v}.pkl", "rb"))
        centre = (H / 2, W / 2)
        cv = {m: M.cv_field_model(p["surfaces"], centre, m) for m in ("const", "affine")}
        model = "affine" if cv["affine"] - cv["const"] >= 0.005 else "const"
        fit = M.fit_field_model(p["surfaces"], centre, model)
        fields[v] = api.FieldResult(fit["theta"], model, centre, None, None, fit["r2"], cv, len(p["surfaces"]))
        print(f"{v}: model {model} CV const {cv['const']:.4f} affine {cv['affine']:.4f}")
        lum[v] = F.luminance(d[f"naip_{v}"])
    D = {v: api.displacement_maps(fields[v], lr) for v in vints}
    errs = {"naive": [], "model": []}
    for a, b in itertools.combinations(vints, 2):
        if MEASURE == "edges":  # robust to crop-colour changes between years (cropland)
            A, B = F.grad_mag(lum[a], 1.5), F.grad_mag(lum[b], 1.5)
        else:
            A = lum[a] - ndimage.gaussian_filter(R.fill_nan(lum[a]), 3)
            B = lum[b] - ndimage.gaussian_filter(R.fill_nan(lum[b]), 3)
        gm = F.grad_mag(lum[a], 1.0) ** 2
        rows = []
        for r in range(0, H - WIN - 2 * RAD + 1, WIN):
            for c in range(0, W - WIN - 2 * RAD + 1, WIN):
                sl = (slice(r, r + WIN + 2 * RAD), slice(c, c + WIN + 2 * RAD))
                res = R.ncc_search(A[sl], B[sl], RAD)
                if res.score < 0.3:
                    continue
                inner = (slice(r + RAD, r + RAD + WIN), slice(c + RAD, c + RAD + WIN))
                w = gm[inner]
                w = w / w.sum()
                pdx = ((D[a][0][inner] - D[b][0][inner]) * w).sum()
                pdy = ((D[a][1][inner] - D[b][1][inner]) * w).sum()
                rows.append((res.dx, res.dy, pdx, pdy))
        x = np.array(rows)
        naive = np.hypot(x[:, 0], x[:, 1])
        model = np.hypot(x[:, 0] - x[:, 2], x[:, 1] - x[:, 3])
        errs["naive"] += naive.tolist()
        errs["model"] += model.tolist()
        print(f"{a} vs {b}: {len(x)} windows | direct NAIP-NAIP |d| rms {np.sqrt((naive**2).mean()):.2f} m | residual after lidar-derived fields rms {np.sqrt((model**2).mean()):.2f} m (median {np.median(model):.2f})")
    print({k: round(float(np.sqrt(np.mean(np.square(v)))), 2) for k, v in errs.items()}, "overall closure RMS (m)")
