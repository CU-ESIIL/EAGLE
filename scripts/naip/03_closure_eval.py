"""Evaluate estimators by (a) block reproducibility and (b) NAIP-vintage closure.

For each site: measure displacement between every pair of NAIP vintages directly (NAIP vs NAIP), then for each lidar-based
estimator compare (shift_a - shift_b) to it. A perfect method gives 0; the "naive" row (shift = 0) shows how inconsistent the
supplied coordinates already are.

Usage: python scripts/naip/03_closure_eval.py CACHE_DIR OUT_JSON SITE_FILE_STEM ...   (e.g. mixed_co crop_ia_5000)
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from eagle_als.coreg import evaluate as E  # noqa: E402
from eagle_als.coreg import features as F  # noqa: E402
from eagle_als.coreg import register as R  # noqa: E402
from eagle_als.coreg.data import LidarRasters  # noqa: E402

RADIUS, BLOCK = 12, 320


def hp(a, s=3.0):
    return a - ndimage.gaussian_filter(R.fill_nan(a), s)


def estimators(naip, lr, X, XE):
    nir, lum = naip[3], F.luminance(naip)
    inten = R.fill_nan(lr.intensity)
    one = lambda a: a[None]  # noqa: E731
    ncc = lambda n, x: R.ncc_search(n[0], x[0], RADIUS)  # noqa: E731
    return {
        "ncc_intensity_nir": (one(nir), one(inten), ncc),
        "ncc_edge_int_lum": (one(F.grad_mag(lum, 1.5)), one(F.grad_mag(inten, 1.5)), ncc),
        "ncc_edge_chm_lum": (one(F.grad_mag(lum, 1.0)), one(F.grad_mag(lr.chm, 1.0)), ncc),
        "mi_chm_nir": (one(nir), one(R.fill_nan(lr.chm)), lambda n, x: R.mi_search(n[0], x[0], RADIUS, step=3)),
        "regression": (naip, X, lambda n, x: R.regression_search(n, x, RADIUS)),
        "regression_edges": (F.naip_edge_stack(naip), XE, lambda n, x: R.regression_search(n, x, RADIUS)),
    }


def pooled(n, x, est):
    blocks = R.block_estimates(n, x, BLOCK, RADIUS, estimator=est)
    if not blocks:
        return None
    res, sd = R.pool_surfaces(blocks, RADIUS)
    a = np.array([[b["dx"], b["dy"]] for b in blocks])
    med = np.median(a, 0)
    return dict(dx=res.dx, dy=res.dy, boot_sd=float(np.hypot(*sd)), block_mad=float(np.hypot(*(1.4826 * np.median(np.abs(a - med), 0)))), score=res.score, n=len(blocks))


if __name__ == "__main__":
    cache, out = sys.argv[1], sys.argv[2]
    results = {}
    for site in sys.argv[3:]:
        d = np.load(Path(cache) / f"{site}.npz")
        lr = LidarRasters(d["dsm"], d["dtm"], d["chm"], d["intensity"], d["density"])
        X, XE = F.lidar_stack(lr), F.lidar_edge_stack(lr)
        keys = sorted(k for k in d.files if k.startswith("naip_"))
        # direct NAIP-vs-NAIP displacement of a relative to b
        pair = {}
        for a, b in E.all_pairs(keys):
            la, lb = hp(F.luminance(d[a]), 2.0), hp(F.luminance(d[b]), 2.0)
            r = pooled(la[None], lb[None], lambda n, x: R.ncc_search(n[0], x[0], RADIUS))
            pair[(a, b)] = (r["dx"], r["dy"], r["boot_sd"])
        results[f"{site}|pairs"] = {f"{a}~{b}": v for (a, b), v in pair.items()}
        naive = E.closure_errors({k: (0.0, 0.0) for k in keys}, {k: v[:2] for k, v in pair.items()})[0]
        print(f"{site}: NAIP vintages mutual displacement RMS (= naive closure error) {E.rms(naive):.2f} m", flush=True)
        per_method = {}
        for k in keys:
            for name, (n, x, est) in estimators(d[k], lr, X, XE).items():
                per_method.setdefault(name, {})[k] = pooled(n, x, est)
        for name, per in per_method.items():
            sh = {k: (v["dx"], v["dy"]) for k, v in per.items() if v}
            errs, _ = E.closure_errors(sh, {k: v[:2] for k, v in pair.items()})
            results[f"{site}|{name}"] = dict(shifts=sh, closure_rms=E.rms(errs), mean_boot_sd=float(np.mean([v["boot_sd"] for v in per.values() if v])), mean_block_mad=float(np.mean([v["block_mad"] for v in per.values() if v])))
            r = results[f"{site}|{name}"]
            print(f"  {name:20s} closure RMS {r['closure_rms']:.2f} m | boot sd {r['mean_boot_sd']:.2f} | block MAD {r['mean_block_mad']:.2f} | " + " ".join(f"{k[5:9]}:({v[0]:+.1f},{v[1]:+.1f})" for k, v in sh.items()), flush=True)
    Path(out).write_text(json.dumps(results, indent=1, default=float))
