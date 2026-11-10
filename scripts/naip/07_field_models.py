"""Fit smooth large-area models to saved class surfaces and score them out of sample.

Models: const (one ground shift + one lean), affine (ground shift and lean both linear in position).
Out-of-sample score: spatial 4-fold CV over 1250 m windows (checkerboard); the held-out blocks are scored by the mean
R^2 (weighted by pixels) of their surfaces evaluated at the model's displacement. Higher = the model predicts where
the NAIP is relative to the lidar in places it has not seen.
Also reports the model's (dx, dy) at the grid centre and the ground-shift range across the area.

Usage: python scripts/naip/07_field_models.py CACHE_DIR STEM VINTAGE [VINTAGE ...]
"""

import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from eagle_als.coreg import model as M  # noqa: E402


def cv(surfs, centre, model, win=1250, folds=4):
    rows = np.array([s["row"] for s in surfs])
    cols = np.array([s["col"] for s in surfs])
    fold = ((rows // win).astype(int) + 2 * (cols // win).astype(int)) % folds
    scores = []
    for f in range(folds):
        tr = [s for s, k in zip(surfs, fold) if k != f]
        te = [s for s, k in zip(surfs, fold) if k == f]
        if len(te) < 5:
            continue
        fit = M.fit_field_model(tr, centre, model)
        st = M.SurfaceStack(te)
        dx, dy = M.field_displacement(fit["theta"], st, centre, model=model)
        scores.append(float((st.n * st.lookup(dx, dy)).sum() / st.n.sum()))
    return float(np.mean(scores))


if __name__ == "__main__":
    cache, stem = sys.argv[1], sys.argv[2]
    for v in sys.argv[3:]:
        d = pickle.load(open(Path(cache) / f"surfaces_{stem}_{v}.pkl", "rb"))
        surfs, (H, W) = d["surfaces"], d["shape"]
        centre = (H / 2, W / 2)
        print(f"== {stem} {v}  sun={d['sun']}  {len(surfs)} surfaces")
        for model in ("const", "affine"):
            fit = M.fit_field_model(surfs, centre, model)
            th = np.round(fit["theta"], 3)
            st = M.SurfaceStack(surfs)
            print(f"  {model:6s} in-sample R2 {fit['r2']:.4f}  CV R2 {cv(surfs, centre, model):.4f}  theta {th.tolist()}")
            if model == "affine":
                # ground shift at the four corners (h = 0)
                for nm, (r, c) in dict(NW=(0, 0), NE=(0, W), SW=(H, 0), SE=(H, W)).items():
                    px, py = (c - centre[1]) / 1000, -(r - centre[0]) / 1000
                    dx = th[0] + th[4] * px + th[5] * py
                    dy = th[1] + th[6] * px + th[7] * py
                    print(f"     ground shift at {nm}: ({dx:+.2f}, {dy:+.2f}) m")
