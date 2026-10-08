"""Stage 4: plots and count tables for the GPS-filtered, ALS-matched LFRDB plots, BEFORE the
missing-label filter (so missing and 'Unclassified' labels are visible).

Reads datasets/raw/LFRDB/derived/lfrdb_gps_als_matched.parquet (stage 3).
Writes datasets/LFRDB_eval/figures/01-04_*.png and catalog/ecosys_counts_matched.csv.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "datasets/raw/LFRDB/derived/lfrdb_gps_als_matched.parquet"
FIG = ROOT / "datasets/LFRDB_eval/figures"
CAT = Path(__file__).parent / "catalog"
TOP = 25
LIFEFORM_COLORS = {"Tree": "#2e7d32", "Shrub": "#b8860b", "Herb": "#7cb342", "Sparse": "#8d6e63", "Other": "#9e9e9e"}


def lifeform_color(x):
    for k, c in LIFEFORM_COLORS.items():
        if isinstance(x, str) and k.lower() in x.lower():
            return c
    return LIFEFORM_COLORS["Other"]


def bar_counts(ax, df, label_col, title):
    ok = df[label_col].notna() & ~df[label_col].astype(str).str.startswith("Unclassified")
    c = df[ok].groupby(label_col).agg(n=("plot_id", "size"), lf=("ecosys_lifeform", lambda s: s.mode().iat[0] if s.notna().any() else None))
    c = c.sort_values("n", ascending=False)
    top = c.head(TOP).iloc[::-1]
    ax.barh(top.index.str.slice(0, 55), top.n, color=[lifeform_color(x) for x in top.lf])
    n_missing = int(df[label_col].isna().sum())
    n_uncl = int(df[label_col].astype(str).str.startswith("Unclassified").sum())
    ax.set_title(f"{title}\n{len(c)} classes with a label; top {min(TOP, len(c))} shown; {n_uncl} 'Unclassified', {n_missing} missing", fontsize=9)
    ax.set_xlabel("plots")
    ax.tick_params(axis="y", labelsize=7)
    return c


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    CAT.mkdir(exist_ok=True)
    d = pd.read_parquet(SRC)

    # 01: community type counts
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.5))
    c = bar_counts(axes[0], d, "ecosys", "Ecological System (primary evaluation label)")
    bar_counts(axes[1], d, "nvc_group", "NVC group (coarser)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=v) for v in LIFEFORM_COLORS.values()]
    axes[1].legend(handles, list(LIFEFORM_COLORS), title="Ecological System lifeform", loc="lower right", fontsize=8)
    fig.suptitle(f"LFRDB plots with GPS location and 3DEP within +/-3 yr (n={len(d)}), before dropping missing labels", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "01_label_counts.png", dpi=130)
    plt.close(fig)
    c.rename_axis("ecosys").reset_index().rename(columns={"lf": "ecosys_lifeform"}).to_csv(CAT / "ecosys_counts_matched.csv", index=False)

    # 02: structure variables
    cover = ["tree_cover_pct", "shrub_cover_pct", "herb_cover_pct"]
    height = ["tree_height_m", "shrub_height_m", "herb_height_m"]
    fig, axes = plt.subplots(2, 4, figsize=(15, 6))
    for ax, col in zip(axes[0, :3], cover):
        ax.hist(d[col].dropna(), bins=np.linspace(0, 100, 26), color="#2e7d32")
        ax.set_title(f"{col}  (n={d[col].notna().sum()})", fontsize=9)
    for ax, col in zip(axes[1, :3], height):
        v = d[col].dropna()
        ax.hist(v, bins=25, color="#b8860b")
        ax.set_title(f"{col}  (n={len(v)})", fontsize=9)
    ax = axes[0, 3]
    cols = cover + height + ["ecosys", "nvc_group", "dominant_species"]
    miss = d[cols].isna().mean().sort_values()
    ax.barh(miss.index, miss.values, color="#9e9e9e")
    ax.set_xlim(0, 1)
    ax.set_title("fraction missing", fontsize=9)
    ax.tick_params(axis="y", labelsize=7)
    ax = axes[1, 3]
    ax.scatter(d.tree_cover_pct, d.shrub_cover_pct, s=3, alpha=0.3)
    ax.set_xlabel("tree_cover_pct", fontsize=8)
    ax.set_ylabel("shrub_cover_pct", fontsize=8)
    ax.set_title("tree vs shrub cover", fontsize=9)
    fig.suptitle("Structure variables (percent cover adjusted for overlap; lifeform height in m). Heights are rarely recorded.", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "02_structure_variables.png", dpi=130)
    plt.close(fig)

    # 03: ALS match
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    d.year.value_counts().sort_index().plot.bar(ax=axes[0], color="#555", title="plot visit year")
    d.collection_year_AWS.value_counts().sort_index().plot.bar(ax=axes[1], color="#1565c0", title="3DEP collection year")
    d.year_diff_AWS.value_counts().sort_index().plot.bar(ax=axes[2], color="#c62828", title="|3DEP year - visit year|")
    d.lfrdb_region.value_counts().plot.bar(ax=axes[3], color="#6a1b9a", title="LANDFIRE region")
    for ax in axes:
        ax.set_xlabel("")
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "03_als_match.png", dpi=130)
    plt.close(fig)

    # 04: map (CONUS panel; AK/HI/insular plots counted in the title)
    fig, ax = plt.subplots(figsize=(8, 5))
    conus = d[d.lon.between(-126, -66) & d.lat.between(24, 50)]
    for r, g in conus.groupby("lfrdb_region"):
        ax.scatter(g.lon, g.lat, s=4, alpha=0.6, label=f"{r} ({len(g)})")
    ax.set_title(f"CONUS plots shown: {len(conus)} of {len(d)} (others are AK, HI, insular areas)", fontsize=9)
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    ax.legend(fontsize=7, markerscale=2)
    fig.tight_layout()
    fig.savefig(FIG / "04_map.png", dpi=130)
    plt.close(fig)
    print("wrote figures to", FIG)


if __name__ == "__main__":
    main()
