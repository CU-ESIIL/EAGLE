"""Compare estimators on every site x NAIP vintage using block-wise estimates (independent 320 m blocks).

For each (site, vintage, method) reports the median NAIP displacement (dx east, dy north, m) over blocks and the robust
spread (MAD*1.48) across blocks = how reproducible the estimate is. Also the closure check against NAIP-vs-NAIP shifts.

Usage: python scripts/naip/02_compare_methods.py CACHE_DIR OUT_JSON [site ...]
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from eagle_als.coreg import features as F  # noqa: E402
from eagle_als.coreg import register as R  # noqa: E402
from eagle_als.coreg.data import LidarRasters  # noqa: E402

RADIUS, BLOCK = 8, 320


def methods(naip, lr, X):
    nir, lum = naip[3], F.luminance(naip)
    ge, gl = F.grad_mag(lum), F.grad_mag(lr.chm)
    one = lambda a: a[None]  # noqa: E731
    return {
        "ncc_intensity_vs_nir": (one(nir), one(R.fill_nan(lr.intensity)), lambda n, x: R.ncc_search(n[0], x[0], RADIUS)),
        "ncc_edges_chm_vs_lum": (one(ge), one(gl), lambda n, x: R.ncc_search(n[0], x[0], RADIUS)),
        "mi_chm_vs_nir": (one(nir), one(R.fill_nan(lr.chm)), lambda n, x: R.mi_search(n[0], x[0], RADIUS, step=2)),
        "regression": (naip, X, lambda n, x: R.regression_search(n, x, RADIUS)),
    }


def summarize(blocks):
    a = np.array([[b["dx"], b["dy"], b["score"]] for b in blocks])
    med = np.median(a[:, :2], 0)
    mad = 1.4826 * np.median(np.abs(a[:, :2] - med), 0)
    return dict(n=len(a), dx=float(med[0]), dy=float(med[1]), mad_x=float(mad[0]), mad_y=float(mad[1]), score=float(np.median(a[:, 2])))


if __name__ == "__main__":
    cache, out = sys.argv[1], sys.argv[2]
    results = {}
    for site in sys.argv[3:] or ["crop_ia", "forest_pa", "mixed_co"]:
        d = np.load(Path(cache) / f"{site}.npz")
        lr = LidarRasters(d["dsm"], d["dtm"], d["chm"], d["intensity"], d["density"])
        X = F.lidar_stack(lr)
        keys = sorted(k for k in d.files if k.startswith("naip_"))
        for k in keys:
            for name, (n, x, est) in methods(d[k], lr, X).items():
                blocks = R.block_estimates(n, x, BLOCK, RADIUS, estimator=est)
                results[f"{site}|{k}|{name}"] = summarize(blocks)
                r = results[f"{site}|{k}|{name}"]
                print(f"{site:10s} {k:16s} {name:22s} ({r['dx']:+.2f},{r['dy']:+.2f}) mad=({r['mad_x']:.2f},{r['mad_y']:.2f}) score={r['score']:.2f}", flush=True)
        # NAIP vs NAIP (newest as reference), same block scheme
        ref = keys[-1] if keys[-1] > keys[0] else keys[0]
        ref = max(keys, key=lambda s: s[5:9])
        for k in keys:
            if k == ref:
                continue
            blocks = R.block_estimates(
                F.luminance(d[ref])[None], F.luminance(d[k])[None], BLOCK, RADIUS, estimator=lambda n, x: R.ncc_search(n[0], x[0], RADIUS)
            )
            results[f"{site}|{ref}|naip_vs|{k}"] = summarize(blocks)
            r = results[f"{site}|{ref}|naip_vs|{k}"]
            print(f"{site:10s} NAIP {ref} vs {k}: ({r['dx']:+.2f},{r['dy']:+.2f}) mad=({r['mad_x']:.2f},{r['mad_y']:.2f})", flush=True)
    Path(out).write_text(json.dumps(results, indent=1))
