"""Analyse saved class surfaces (04_shift_field.py): windowed (d0, lean) fits and large-scale models.

Windowed: every `--win` m window (step = win) gets its own (dx, dy, tx, ty) + block-bootstrap sd.
Global models over the whole area, compared by leave-one-window-out error of the windowed ground shift:
  const   : one (dx, dy) for the whole area
  plane   : (dx, dy) = linear function of position
  window  : independent windows (no pooling)

Usage: python scripts/naip/05_field_analysis.py CACHE_DIR STEM VINTAGE WIN_M OUT_JSON
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from eagle_als.coreg import model as M  # noqa: E402


def window_fit(surfs, n_boot=20, seed=0):
    rng = np.random.default_rng(seed)
    f = M.fit_lean_model(surfs, fit_lean=True)
    keys = sorted({(s["row"], s["col"]) for s in surfs})
    by = {}
    for s in surfs:
        by.setdefault((s["row"], s["col"]), []).append(s)
    boots = []
    for _ in range(n_boot):
        sub = [s for k in (keys[i] for i in rng.integers(0, len(keys), len(keys))) for s in by[k]]
        g = M.fit_lean_model(sub, fit_lean=True)
        boots.append((g["dx"], g["dy"], g["tx"], g["ty"]))
    b = np.array(boots)
    f["sd_xy"] = float(np.hypot(*b[:, :2].std(0)))
    f["sd_t"] = float(np.hypot(*b[:, 2:].std(0)))
    f["n_blocks"] = len(keys)
    return f


if __name__ == "__main__":
    cache, stem, vintage, win, out = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
    d = pickle.load(open(Path(cache) / f"surfaces_{stem}_{vintage}.pkl", "rb"))
    H, W = d["shape"]
    surfs = d["surfaces"]
    rows = []
    for r0 in range(0, H - win + 1, win):
        for c0 in range(0, W - win + 1, win):
            sub = [s for s in surfs if r0 <= s["row"] < r0 + win and c0 <= s["col"] < c0 + win]
            if len({(s["row"], s["col"]) for s in sub}) < 4:
                continue
            f = window_fit(sub)
            f.update(row=r0 + win / 2, col=c0 + win / 2)
            rows.append(f)
            print(f"{vintage} win r{int(f['row'])} c{int(f['col'])}: d=({f['dx']:+.2f},{f['dy']:+.2f}) t=({f['tx']:+.3f},{f['ty']:+.3f}) r2={f['r2']:.3f} sd={f['sd_xy']:.2f}/{f['sd_t']:.3f} nb={f['n_blocks']}", flush=True)
    a = np.array([[r["dx"], r["dy"], r["tx"], r["ty"], r["row"], r["col"], r["sd_xy"]] for r in rows])
    good = a[:, 6] < 1.5
    print(f"{good.sum()}/{len(a)} windows with bootstrap sd < 1.5 m")
    g = a[good]
    print("window spread of ground shift (std dx, dy):", g[:, :2].std(0).round(2), " range dx", g[:, 0].min().round(2), g[:, 0].max().round(2), " dy", g[:, 1].min().round(2), g[:, 1].max().round(2))
    # planar fit of ground shift and lean over position (km)
    A = np.c_[np.ones(len(g)), (g[:, 5] - W / 2) / 1000, (g[:, 4] - H / 2) / 1000]
    for i, nm in enumerate(["dx", "dy", "tx", "ty"]):
        c, *_ = np.linalg.lstsq(A, g[:, i], rcond=None)
        res = g[:, i] - A @ c
        print(f"  plane fit {nm}: {c[0]:+.3f} + {c[1]:+.3f}*x_km + {c[2]:+.3f}*y_km  resid sd {res.std():.3f} (raw sd {g[:, i].std():.3f})")
    Path(out).write_text(json.dumps(dict(vintage=vintage, win=win, windows=rows), indent=1, default=float))
