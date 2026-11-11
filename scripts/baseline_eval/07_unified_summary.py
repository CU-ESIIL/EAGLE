"""Unified results table: one row per task and class set, one column per feature type (model).

    python scripts/baseline_eval/07_unified_summary.py

Reads results/*/summary.csv and writes results/summary.md. Score = bal_acc (multiclass), mean_auroc (multilabel),
mean_r2 (regression). Also called at the end of 02_linear_probe.py and 06_location_baseline.py.
"""

from pathlib import Path

import pandas as pd

RESULTS = Path(__file__).resolve().parent / "results"
SUPERSEDED = {"alphaearth"}  # old NLCD-only run (disc + centre); alphaearth_center replaces it
PRIMARY = dict(multiclass="bal_acc", multilabel="mean_auroc", regression="mean_r2")


def build():
    cols = {}
    for f in sorted(RESULTS.glob("*/summary.csv")):
        if f.parent.name in SUPERSEDED:
            continue
        df = pd.read_csv(f)
        if df.empty or "kind" not in df:
            continue
        df["score"] = [r[PRIMARY[r["kind"]]] for _, r in df.iterrows()]
        df["min_count"] = df["min_count"].fillna(0).astype(int)  # regression has no class set
        multi = df["pooling"].nunique() > 1
        for pooling, g in df.groupby("pooling"):
            name = f"{f.parent.name}/{pooling}" if multi else f.parent.name
            cols[name] = g.set_index(["task", "min_count"])["score"]
    table = pd.DataFrame(cols)
    kinds = {}
    for f in RESULTS.glob("*/summary.csv"):
        if f.parent.name in SUPERSEDED:
            continue
        d = pd.read_csv(f)
        if "kind" in d:
            kinds.update(dict(zip(d["task"], d["kind"])))
    table.insert(0, "score", [PRIMARY[kinds[t]] for t, _ in table.index])
    table = table.reset_index().rename(columns={"min_count": "min_count (0 = n/a)"})
    order = {t: i for i, t in enumerate(pd.read_csv(next(RESULTS.glob("*/summary.csv")))["task"].unique())}
    table = table.sort_values(["task", "min_count (0 = n/a)"], key=lambda s: s.map(order) if s.name == "task" else s,
                              ascending=[True, False])
    return table


def main():
    t = build()
    lines = ["# Baseline results: score per task and feature type", "",
             "Test rows of the spatial hold-out. `score` names the metric (bal_acc = balanced accuracy, mean_auroc, "
             "mean_r2); `min_count` is the class-set scenario (20 = classes with >= 20 rows in train and test, "
             "1 = all). Blank = not run for that feature type. Per-model detail: `results/<model>/summary.md`.", "",
             t.round(3).astype(object).where(t.notna(), "").to_markdown(index=False), ""]
    (RESULTS / "summary.md").write_text("\n".join(lines))
    print(f"[summary] wrote {RESULTS / 'summary.md'} ({len(t)} rows, {t.shape[1] - 3} feature types)")


if __name__ == "__main__":
    main()
