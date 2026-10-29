"""Samples per class, taxonomy level and split for the VegBench eval table (lidar within +/-3 yr).

Splits are VegBench's own: train, val, test/strat (spatially stratified) and test/acc (accuracy assessment).
Counts for the whole benchmark (before the lidar match) are written next to them for comparison.

Writes:
  catalog/class_counts_by_split.csv     one row per (level, class): rows per split, eval table and full benchmark
  catalog/class_counts_summary.csv      one row per level: classes per split, and how many are scoreable
  datasets/VegBench_eval/figures/02_class_counts_rank.png     rank-abundance curves per level, each split ranked on its own
  datasets/VegBench_eval/figures/03_class_counts_coarse.png   per-class stacked counts, 6 coarsest levels
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "datasets/VegBench_eval/vegbench_eval.parquet"
FULL = ROOT / "datasets/raw/vegbench/benchmark_splits/VegBench/VegBench.csv"
CAT = Path(__file__).parent / "catalog"
FIG = ROOT / "datasets/VegBench_eval/figures"
LEVELS = ["formation_class", "formation_subclass", "formation", "division", "macrogroup", "group", "alliance",
          "association"]
GROUPS = ["train", "val", "test/strat", "test/acc"]
COLORS = {"train": "#2a6f97", "val": "#e9a03b", "test/strat": "#5aa469", "test/acc": "#a4437a"}
MIN_N = 20  # "scoreable": at least this many rows in train and in each test set (strict), or in train and in the two test sets combined (loose)


def group(d):
    return np.where(d.split == "test", "test/" + d.test_type.fillna(""), d.split)


def counts(d, lv):
    c = d.assign(g=group(d)).groupby([f"survey_{lv}", "g"]).size().unstack(fill_value=0)
    return c.reindex(columns=GROUPS, fill_value=0)


def main():
    ev = pd.read_parquet(EVAL)
    full = pd.read_csv(FULL, low_memory=False)
    rows, summ = [], []
    tabs = {}
    for lv in LEVELS:
        ce, cf = counts(ev, lv), counts(full, lv)
        tabs[lv] = ce
        t = ce.add_prefix("n_").join(cf.add_prefix("full_n_"), how="outer").fillna(0).astype(int)
        t["n_total"] = t[[f"n_{g}" for g in GROUPS]].sum(axis=1)
        t.insert(0, "level", lv)
        rows.append(t.reset_index(names="class"))
        pres = ce.gt(0)
        loose = (ce.train >= MIN_N) & ((ce["test/strat"] + ce["test/acc"]) >= MIN_N)
        scoreable = (ce.train >= MIN_N) & (ce["test/strat"] >= MIN_N) & (ce["test/acc"] >= MIN_N)
        summ.append(dict(
            level=lv, n_rows_labeled=int(ce.values.sum()), n_classes_full_benchmark=int((cf.sum(axis=1) > 0).sum()),
            n_classes=len(ce), **{f"classes_in_{g}": int(pres[g].sum()) for g in GROUPS},
            test_classes_not_in_train=int((pres[["test/strat", "test/acc"]].any(axis=1) & ~pres.train).sum()),
            rows_of_those=int(ce.loc[pres[["test/strat", "test/acc"]].any(axis=1) & ~pres.train,
                                    ["test/strat", "test/acc"]].values.sum()),
            **{f"classes_ge{MIN_N}_train_and_each_test": int(scoreable.sum())},
            rows_in_those=int(ce[scoreable].values.sum()),
            **{f"classes_ge{MIN_N}_train_and_ge{MIN_N}_test_total": int(loose.sum())},
            rows_in_those_loose=int(ce[loose].values.sum())))
    allc = pd.concat(rows, ignore_index=True)
    allc.to_csv(CAT / "class_counts_by_split.csv", index=False)
    s = pd.DataFrame(summ)
    s.to_csv(CAT / "class_counts_summary.csv", index=False)
    print(s.to_string(index=False))

    # rank-abundance
    fig, axs = plt.subplots(2, 4, figsize=(15, 6.5), sharey=True)
    for ax, lv in zip(axs.ravel(), LEVELS):
        ce = tabs[lv]
        for g in GROUPS:
            v = np.sort(ce[g].values[ce[g].values > 0])[::-1]
            ax.plot(np.arange(1, len(v) + 1), v, color=COLORS[g], lw=1.5, label=f"{g} ({len(v)})")
        ax.axhline(MIN_N, color="grey", ls=":", lw=1)
        ax.set_yscale("log"), ax.set_xscale("log" if len(ce) > 100 else "linear")
        ax.set_title(f"{lv} ({len(ce)} classes)", fontsize=10), ax.grid(alpha=0.3)
        ax.set_xlabel("class rank within split")
    for ax in axs[:, 0]:
        ax.set_ylabel("rows in class")
    [a.legend(fontsize=7) for a in axs.ravel()]
    fig.suptitle(f"VegBench eval table (lidar within 3 yr): rows per class, classes ranked within each split (legend: classes present); dotted line = {MIN_N}", y=0.995)
    fig.tight_layout()
    fig.savefig(FIG / "02_class_counts_rank.png", dpi=150)
    plt.close(fig)

    # stacked bars for coarse levels
    show = LEVELS[:6]
    fig, axs = plt.subplots(2, 3, figsize=(17, 11), gridspec_kw={"height_ratios": [0.8, 1.6]})
    sizes = {lv: len(tabs[lv]) for lv in show}
    axs = axs.ravel()
    for ax, lv in zip(axs, show):
        ce = tabs[lv].copy()
        ce = ce.loc[ce.sum(axis=1).sort_values().index]
        left = np.zeros(len(ce))
        for g in GROUPS:
            ax.barh(range(len(ce)), ce[g], left=left, color=COLORS[g], label=g)
            left += ce[g].values
        lab = [str(c)[:38] for c in ce.index]
        ax.set_yticks(range(len(ce)), lab, fontsize=7 if len(ce) < 20 else 6)
        ax.set_title(f"{lv} ({sizes[lv]} classes)", fontsize=10)
        ax.set_xlabel("rows")
    axs[0].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "03_class_counts_coarse.png", dpi=130)


if __name__ == "__main__":
    main()
