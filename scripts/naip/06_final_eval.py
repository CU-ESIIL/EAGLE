"""Final method comparison on the 1.5 km test sites, all NAIP vintages.

Variants (shift = displacement of NAIP rel. to lidar, m; pooled over 320 m blocks unless stated):
  naive               (0, 0): trust the supplied coordinates
  ncc_intensity_nir   NCC lidar intensity vs NAIP NIR
  ncc_edges           NCC |grad| lidar intensity vs |grad| NAIP luminance
  regr_raw            lidar features -> NAIP bands regression (no edges, no shadows)
  regr_edges          + edge features/targets
  regr_shadow         + fitted cast-shadow features
  full                + height-class surfaces with relief lean (api.estimate), ground-plane shift reported

Metrics: (1) closure RMS: lidar-derived (d_a - d_b) vs the directly measured NAIP-vs-NAIP displacement on open ground
(<0.5 m CHM, >10 m from anything taller than 2 m), over all vintage pairs. (2) bootstrap sd over blocks. (3) known-shift
recovery on the NAIP of the newest vintage displaced by (+2.37, -1.61) m.

Usage: python scripts/naip/06_final_eval.py CACHE_DIR SITE OUT_JSON
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from eagle_als.coreg import api  # noqa: E402
from eagle_als.coreg import evaluate as E  # noqa: E402
from eagle_als.coreg import features as F  # noqa: E402
from eagle_als.coreg import model as M  # noqa: E402
from eagle_als.coreg import register as R  # noqa: E402
from eagle_als.coreg.data import LidarRasters  # noqa: E402

RAD, BLOCK = 12, 320


def hp(a, s=3.0):
    return a - ndimage.gaussian_filter(R.fill_nan(a), s)


def pooled(Y, X, est=None):
    blocks = R.block_estimates(Y, X, BLOCK, RAD, estimator=est)
    r, sd = R.pool_surfaces(blocks, RAD)
    return dict(dx=r.dx, dy=r.dy, boot_sd=float(np.hypot(*sd)))


def ground_pairs(d, lr, keys):
    chm = np.nan_to_num(lr.chm)
    near_tall = ndimage.binary_dilation(chm > 2, iterations=10)
    w = ((chm < 0.5) & ~near_tall).astype(float)
    out = {}
    for a, b in E.all_pairs(keys):
        la, lb = hp(F.luminance(d[a]))[None], hp(F.luminance(d[b]))[None]
        blocks = R.block_estimates(la, lb, BLOCK, RAD, weight=w, estimator=lambda n, x, w_: R.ncc_search(n[0], x[0], RAD, mask=w_))
        r, sd = R.pool_surfaces(blocks, RAD)
        out[(a, b)] = (r.dx, r.dy)
    return out, float(w.mean())


def variants(naip, lr, sun=None):
    X0 = F.lidar_stack(lr)
    nir, lum = naip[3], F.luminance(naip)
    inten = R.fill_nan(lr.intensity)
    ncc = lambda n, x: R.ncc_search(n[0], x[0], RAD)  # noqa: E731
    res = {}
    res["ncc_intensity_nir"] = pooled(nir[None], inten[None], ncc)
    res["ncc_edges"] = pooled(F.grad_mag(lum, 1.5)[None], F.grad_mag(inten, 1.5)[None], ncc)
    res["regr_raw"] = pooled(naip, X0)
    Y, X = api.build(naip, lr)
    res["regr_edges"] = pooled(Y, X)
    if sun is None:
        az, el, _ = M.fit_sun(Y, lr.dsm, X, radius=8, window=500) if api.tall_fraction(lr.chm) > 0.05 else (None, None, None)
        sun = (az, el) if az is not None else None
    if sun is not None:
        Ys, Xs = api.build(naip, lr, sun)
        res["regr_shadow"] = pooled(Ys, Xs)
    else:
        res["regr_shadow"] = res["regr_edges"]
    r = api.estimate(naip, lr, sun=sun, n_boot=20)
    res["full"] = dict(dx=r.dx, dy=r.dy, boot_sd=r.sd, tx=r.tx, ty=r.ty, sun=sun, r2=r.r2)
    return res


if __name__ == "__main__":
    cache, site, out = sys.argv[1:4]
    d = np.load(Path(cache) / f"{site}.npz")
    lr = LidarRasters(d["dsm"], d["dtm"], d["chm"], d["intensity"], d["density"])
    keys = sorted(k for k in d.files if k.startswith("naip_"))
    results = dict(site=site, vintages={})
    t0 = time.time()
    pairs, open_frac = ground_pairs(d, lr, keys)
    results["open_ground_fraction"] = open_frac
    results["pairs"] = {f"{a}~{b}": v for (a, b), v in pairs.items()}
    naive = E.closure_errors({k: (0.0, 0.0) for k in keys}, pairs)[0]
    results["closure_rms"] = {"naive": E.rms(naive)}
    print(f"{site}: pairs done {time.time() - t0:.0f}s; naive closure RMS {E.rms(naive):.2f} m (open ground {open_frac:.2f})", flush=True)
    for k in keys:
        results["vintages"][k] = variants(d[k], lr)
        print(k, {m: (round(v["dx"], 2), round(v["dy"], 2)) for m, v in results["vintages"][k].items()}, f"{time.time() - t0:.0f}s", flush=True)
        json.dump(results, open(out, "w"), indent=1, default=float)
    for m in results["vintages"][keys[0]]:
        sh = {k: (results["vintages"][k][m]["dx"], results["vintages"][k][m]["dy"]) for k in keys}
        results["closure_rms"][m] = E.rms(E.closure_errors(sh, pairs)[0])
    # known-shift recovery on newest vintage
    newest = max(keys, key=lambda s: s[5:9])
    dx, dy = 2.37, -1.61
    shifted = np.stack([ndimage.shift(b, (-dy, dx), order=3, mode="nearest") for b in d[newest]])
    base, moved = results["vintages"][newest], variants(shifted, lr, sun=results["vintages"][newest]["full"]["sun"])
    results["known_shift"] = {m: (moved[m]["dx"] - base[m]["dx"] - dx, moved[m]["dy"] - base[m]["dy"] - dy) for m in moved}
    print("closure RMS", results["closure_rms"])
    print("known-shift error (applied +2.37,-1.61)", results["known_shift"])
    json.dump(results, open(out, "w"), indent=1, default=float)
