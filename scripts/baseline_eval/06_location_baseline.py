"""Location-only baseline: lat/lon -> random Fourier features -> the same linear probes as the other models.

    python scripts/baseline_eval/06_location_baseline.py                 # all tasks
    python scripts/baseline_eval/06_location_baseline.py --tasks nlcd --n-jobs 4

Shows how much of each task is geography (the same spatial hold-out as every other model). Points are mapped to
the unit sphere and projected on random directions at several length scales (km), so the linear probe sees
multi-scale position. AEF minus this baseline = what the embedding adds beyond position. Needs no lidar or imagery.
Writes scripts/baseline_eval/results/<model>/{summary.csv,per_target.csv,summary.md}.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/baseline_eval"))
from eagle_bench import probes  # noqa: E402
from eagle_bench.tasks import TASKS, load_task  # noqa: E402

EARTH_KM = 6371.0
HEADLINE = dict(multiclass=["bal_acc", "macro_f1", "acc", "chance", "n_classes"],
                multilabel=["mean_auroc", "mean_ap", "mean_prevalence", "n_species"],
                regression=["mean_r2", "mean_spearman", "n_targets"])


def rff(lat, lon, scales_km, n_per_scale, seed=0):
    """Random Fourier features of the position on the sphere: cos/sin(w . xyz + b), |w| ~ 1/scale."""
    lat, lon = np.radians(lat), np.radians(lon)
    xyz = np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], 1) * EARTH_KM
    rng = np.random.default_rng(seed)
    feats = []
    for s in scales_km:
        W = rng.normal(size=(3, n_per_scale)) / s
        b = rng.uniform(0, 2 * np.pi, n_per_scale)
        feats.append(np.cos(xyz @ W + b))
    return np.concatenate(feats, 1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=list(TASKS))
    ap.add_argument("--min-counts", nargs="*", type=int, default=[20, 1])
    ap.add_argument("--scales-km", nargs="*", type=float, default=[10, 30, 100, 300, 1000, 3000])
    ap.add_argument("--n-per-scale", type=int, default=32)
    ap.add_argument("--model", default="location_rff")
    ap.add_argument("--n-jobs", type=int, default=-1)
    a = ap.parse_args()
    out = ROOT / "scripts/baseline_eval/results" / a.model
    out.mkdir(parents=True, exist_ok=True)
    from importlib import import_module
    write_markdown = import_module("02_linear_probe").write_markdown

    summaries, per_target = [], []
    for name in a.tasks:
        task = load_task(name)
        X = rff(task.df["lat"].to_numpy(), task.df["lon"].to_numpy(), a.scales_km, a.n_per_scale)
        X[~np.isfinite(task.df[["lat", "lon"]].to_numpy()).all(1)] = np.nan
        for min_count in ([None] if task.kind == "regression" else a.min_counts):
            s, per = probes.evaluate(task, X, n_jobs=a.n_jobs, min_count=min_count)
            s.update(pooling="location", frac_year_clipped=task.meta["frac_year_clipped"], description=task.description)
            per.insert(1, "pooling", "location")
            per.insert(2, "min_count", min_count)
            summaries.append(s)
            per_target.append(per)
            print(f"[loc] {name} (min_count={min_count}): " + ", ".join(
                f"{k}={s[k]:.3f}" for k in HEADLINE[task.kind] if k in s), flush=True)
            pd.DataFrame(summaries).to_csv(out / "summary.csv", index=False)
            pd.concat(per_target, ignore_index=True).to_csv(out / "per_target.csv", index=False)
    write_markdown(pd.DataFrame(summaries), out / "summary.md", a.model)
    import_module("07_unified_summary").main()


if __name__ == "__main__":
    main()
