"""Linear-probe the extracted AlphaEarth embeddings on every benchmark task and write the results.

    python scripts/baseline_eval/02_linear_probe.py                          # all tasks, both poolings
    python scripts/baseline_eval/02_linear_probe.py --tasks nlcd ofo_trees --pooling disc
    python scripts/baseline_eval/02_linear_probe.py --min-counts 20        # only the ">= 20 per split" class set

Class-set scenarios (--min-counts, default 20 and 1): a class or species is scored only if it has at least this
many rows in each of train and test. Run on the full tables, 1 = all classes, 20 = the stable ones. Regression
tasks have no class set and run once (min_count blank).

Reads datasets/raw/aef/points/ (from 01_extract_aef.py). Writes, in scripts/baseline_eval/results/<model>/:
    summary.csv     one row per task and pooling: headline metrics, class/species counts, rows used
    per_target.csv  per class (F1), species (AUROC, AP) or regression target (R2, Spearman, RMSE)
    summary.md      the same headline table for reading
Re-use for another embedding: write a points table with the same `key` + embedding columns, or pass your own
matrices to `eagle_bench.probes.evaluate(task, X)`.

Lidar embeddings / features keyed by cookie id (e.g. ForPT, from 04_embed_forpt.py): pass a directory of parquet
shards with an `als_site_id` column; every other numeric column except `n_input` is used as a feature (minus --exclude; NaNs in rows with a cookie are
filled with the train median), and rows of a
task are matched on `task.df.als_site_id` (rows without a cookie count as n_no_embedding):

    python scripts/baseline_eval/02_linear_probe.py --embeddings $BENCH/embeddings/forpt/r50_z --model forpt_r50_z
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
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
    ap.add_argument("--pooling", nargs="*", default=["disc", "center"], choices=["disc", "center"])  # AEF only
    ap.add_argument("--min-counts", nargs="*", type=int, default=[20, 1])
    ap.add_argument("--model", default="alphaearth")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--embeddings", default=None, help="dir of parquet shards keyed by als_site_id (lidar features)")
    ap.add_argument("--exclude", default=None, help="regex of feature columns to leave out (with --embeddings)")
    a = ap.parse_args()
    out = ROOT / "scripts/baseline_eval/results" / a.model
    out.mkdir(parents=True, exist_ok=True)

    if a.embeddings:
        emb = pd.concat([pd.read_parquet(f) for f in sorted(Path(a.embeddings).glob("*.parquet"))], ignore_index=True)
        emb = emb.drop_duplicates("als_site_id").set_index("als_site_id")
        emb = emb[[c for c in emb.columns if c != "n_input" and pd.api.types.is_numeric_dtype(emb[c])
                   and not (a.exclude and re.search(a.exclude, c))]]
        emb = emb.loc[:, emb.notna().any()]  # all-NaN columns carry nothing
        a.pooling = ["cookie"]
        print(f"[probe] {len(emb)} cookie embeddings, {emb.shape[1]} features", flush=True)
    else:
        points = aef.load_points(a.points)
        print(f"[probe] {len(points)} extracted points", flush=True)
    summaries, per_target = [], []
    for name in a.tasks:
        task = load_task(name)
        scenarios = [None] if task.kind == "regression" else a.min_counts
        for pooling, min_count in [(p, m) for p in a.pooling for m in scenarios]:
            if a.embeddings:
                ids = task.df["als_site_id"] if "als_site_id" in task.df else pd.Series(index=task.df.index, dtype=object)
                X = emb.reindex(ids).to_numpy(np.float32)
                X = impute_train_median(X, ~task.df["is_test"].to_numpy())
            else:
                X = aef.matrix_for(task.df, points, pooling)
            s, per = probes.evaluate(task, X, n_jobs=a.n_jobs, min_count=min_count)
            s.update(pooling=pooling, frac_year_clipped=task.meta["frac_year_clipped"], description=task.description)
            per.insert(1, "pooling", pooling)
            per.insert(2, "min_count", min_count)
            summaries.append(s)
            per_target.append(per)
            print(f"[probe] {name} ({pooling}, min_count={min_count}): " + ", ".join(f"{k}={s[k]:.3f}" for k in HEADLINE[task.kind] if k in s),
                  flush=True)
            pd.DataFrame(summaries).to_csv(out / "summary.csv", index=False)  # written as we go
            pd.concat(per_target, ignore_index=True).to_csv(out / "per_target.csv", index=False)
    write_markdown(pd.DataFrame(summaries), out / "summary.md", a.model)


def impute_train_median(X, is_train):
    """Rows with a cookie but undefined metrics (e.g. lidR zentropy on flat water, terrain without ground points):
    fill each NaN with the median of that column over the task's train rows. Rows with no cookie (all NaN) stay NaN
    and are excluded and counted as n_no_embedding."""
    has = np.isfinite(X).any(1)
    med = np.nanmedian(np.where((has & is_train)[:, None], X, np.nan), axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    X = X.copy()
    fill = has[:, None] & ~np.isfinite(X)
    X[fill] = np.broadcast_to(med, X.shape)[fill]
    return X


def write_markdown(df, path, model):
    lines = [f"# {model}: linear probe results", "",
             "Test rows only (the datasets' spatial hold-out). `frac clipped` = share of rows whose survey year is "
             "outside the imagery years and uses the nearest available year.", ""]
    for kind, cols in HEADLINE.items():
        sub = df[df["kind"] == kind]
        if sub.empty:
            continue
        keep = ["task", "pooling", "min_count", "n_train", "n_test", "n_no_embedding", "frac_year_clipped"] + cols
        lines += [f"## {kind}", "", sub[[c for c in keep if c in sub]].round(3).to_markdown(index=False), ""]
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
