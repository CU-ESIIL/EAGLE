"""Linear-probe the extracted AlphaEarth embeddings on every benchmark task and write the results.

    python scripts/baseline_eval/02_linear_probe.py                          # all tasks, both poolings
    python scripts/baseline_eval/02_linear_probe.py --tasks nlcd ofo_trees --pooling disc

Reads datasets/raw/aef/points/ (from 01_extract_aef.py). Writes, in scripts/baseline_eval/results/<model>/:
    summary.csv     one row per task and pooling: headline metrics, class/species counts, rows used
    per_target.csv  per class (F1), species (AUROC, AP) or regression target (R2, Spearman, RMSE)
    summary.md      the same headline table for reading
Re-use for another embedding: write a points table with the same `key` + embedding columns, or pass your own
matrices to `eagle_bench.probes.evaluate(task, X)`.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from eagle_bench import aef, probes  # noqa: E402
from eagle_bench.tasks import TASKS, load_task  # noqa: E402

HEADLINE = dict(multiclass=["bal_acc", "macro_f1", "acc", "chance", "n_classes"],
                multilabel=["mean_auroc", "mean_ap", "mean_prevalence", "n_species"],
                regression=["mean_r2", "mean_spearman", "n_targets"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=list(TASKS))
    ap.add_argument("--points", default=str(aef.CACHE_DIR / "points"))
    ap.add_argument("--pooling", nargs="*", default=["disc", "center"], choices=["disc", "center"])
    ap.add_argument("--model", default="alphaearth")
    ap.add_argument("--n-jobs", type=int, default=-1)
    a = ap.parse_args()
    out = ROOT / "scripts/baseline_eval/results" / a.model
    out.mkdir(parents=True, exist_ok=True)

    points = aef.load_points(a.points)
    print(f"[probe] {len(points)} extracted points", flush=True)
    summaries, per_target = [], []
    for name in a.tasks:
        task = load_task(name)
        for pooling in a.pooling:
            X = aef.matrix_for(task.df, points, pooling)
            s, per = probes.evaluate(task, X, n_jobs=a.n_jobs)
            s.update(pooling=pooling, frac_year_clipped=task.meta["frac_year_clipped"], description=task.description)
            per.insert(1, "pooling", pooling)
            summaries.append(s)
            per_target.append(per)
            print(f"[probe] {name} ({pooling}): " + ", ".join(f"{k}={s[k]:.3f}" for k in HEADLINE[task.kind] if k in s),
                  flush=True)
            pd.DataFrame(summaries).to_csv(out / "summary.csv", index=False)  # written as we go
            pd.concat(per_target, ignore_index=True).to_csv(out / "per_target.csv", index=False)
    write_markdown(pd.DataFrame(summaries), out / "summary.md", a.model)


def write_markdown(df, path, model):
    lines = [f"# {model}: linear probe results", "",
             "Test rows only (the datasets' spatial hold-out). `frac clipped` = share of rows whose survey year is "
             "outside the imagery years and uses the nearest available year.", ""]
    for kind, cols in HEADLINE.items():
        sub = df[df["kind"] == kind]
        if sub.empty:
            continue
        keep = ["task", "pooling", "n_train", "n_test", "n_no_embedding", "frac_year_clipped"] + cols
        lines += [f"## {kind}", "", sub[[c for c in keep if c in sub]].round(3).to_markdown(index=False), ""]
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
