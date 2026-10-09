"""Stage 5: drop records without a usable primary label, make the 50% test hold-out, export.

Primary evaluation column: `ecosys` (LANDFIRE/NatureServe Ecological System). Rows where it is
null or an 'Unclassified ...' fallback (a lifeform bucket, not a habitat type) are dropped.
Structure columns (cover, height) are NOT used to drop rows; they stay null where not recorded.

Split: ~50% of rows held out, assigned by 0.1-degree lat/lon blocks (spatial hold-out; plots
from one survey sit close together, so a random plot split would leak). Blocks are shuffled with
SEED (same seed as the BBS and butterfly splits) and added to the test set while that moves the
test share closer to 50%.

Reads datasets/raw/LFRDB/derived/lfrdb_gps_als_matched.parquet (stage 3).
Writes datasets/LFRDB_eval/lfrdb_eval.parquet, figures/05_split_*.png,
catalog/filter_counts.csv and ecosys_split_counts.csv (plots per class and split).
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "datasets/raw/LFRDB/derived/lfrdb_gps_als_matched.parquet"
OUT = ROOT / "datasets/LFRDB_eval"
CAT = Path(__file__).parent / "catalog"
SEED = 20260909
TEST_FRACTION = 0.5
MIN_PER_SPLIT = 20  # for the "usable classes" count in the summary only

COLUMNS = [
    "plot_id", "lat", "lon", "year", "month",
    "ecosys_code", "ecosys", "ecosys_lifeform", "nvc_group_code", "nvc_group",
    "tree_cover_pct", "shrub_cover_pct", "herb_cover_pct",
    "tree_height_m", "shrub_height_m", "herb_height_m",
    "dominant_lifeform", "dominant_species", "evt_method",
    "source_id", "source_agency", "lfrdb_region", "loc_method", "n_decimals",
    "product_name_AWS", "collection_year_AWS", "year_diff_AWS", "als_site_id", "test_split",
]


def spatial_split(d):
    block = (np.floor(d.lat)*10).astype(int) * 1000 + (np.floor(d.lon)*10).astype(int)
    sizes = block.value_counts()
    order = np.sort(sizes.index.to_numpy())  # sorted, writable, so the shuffle depends only on SEED
    np.random.default_rng(SEED).shuffle(order)
    target = TEST_FRACTION * len(d)
    test, n = set(), 0
    for b in order:
        if abs(n + sizes[b] - target) < abs(n - target):
            test.add(b)
            n += sizes[b]
    return block.isin(test)


def main():
    d = pd.read_parquet(SRC)
    counts = pd.read_csv(CAT / "filter_counts_stage3.csv")[["step", "n_plots"]]
    rows = [tuple(r) for r in counts.itertuples(index=False)]

    d = d[d.ecosys.notna()]
    rows.append(("8. ecosys label not missing", len(d)))
    d = d[~d.ecosys.str.startswith("Unclassified")]
    rows.append(("9. ecosys is not an 'Unclassified ...' fallback", len(d)))
    fc = pd.DataFrame(rows, columns=["step", "n_plots"])
    fc["dropped"] = (-fc.n_plots.diff()).fillna(0).astype(int)
    fc.to_csv(CAT / "filter_counts.csv", index=False)

    d = d.copy()
    d["test_split"] = spatial_split(d).values
    d = d[COLUMNS].sort_values("plot_id").reset_index(drop=True)
    OUT.mkdir(exist_ok=True)
    d.to_parquet(OUT / "lfrdb_eval.parquet", index=False)

    c = d.pivot_table(index=["ecosys_code", "ecosys", "ecosys_lifeform"], columns="test_split", values="plot_id", aggfunc="size", fill_value=0)
    c.columns = ["n_train", "n_test"]
    c["n"] = c.n_train + c.n_test
    c = c.sort_values("n", ascending=False).reset_index()
    c.to_csv(OUT / "ecosys_split_counts.csv", index=False)

    print(fc.to_string(index=False))
    print(f"\nfinal: {len(d)} plots, {d.ecosys.nunique()} Ecological Systems, {d.nvc_group.nunique()} NVC groups, {d.plot_id.nunique()} unique ids")
    print("test_split:", d.test_split.value_counts().to_dict(), f"({d.test_split.mean():.3f} test)")
    both = ((c.n_train >= MIN_PER_SPLIT) & (c.n_test >= MIN_PER_SPLIT)).sum()
    print(f"classes with >= {MIN_PER_SPLIT} plots in both splits: {both}; classes in only one split: {((c.n_train == 0) | (c.n_test == 0)).sum()}")
    print("non-null structure:", d[["tree_cover_pct", "shrub_cover_pct", "herb_cover_pct", "tree_height_m", "shrub_height_m", "herb_height_m"]].notna().sum().to_dict())

    # figures
    fig, ax = plt.subplots(figsize=(8, 5))
    for flag, color, name in [(False, "#1565c0", "train"), (True, "#c62828", "test")]:
        g = d[(d.test_split == flag) & d.lon.between(-126, -66) & d.lat.between(24, 50)]
        ax.scatter(g.lon, g.lat, s=4, alpha=0.6, color=color, label=f"{name} ({int((d.test_split == flag).sum())} total)")
    ax.set_title("Train/test split by 0.1-degree block (CONUS shown)", fontsize=9)
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    ax.legend(fontsize=8, markerscale=2)
    fig.tight_layout()
    fig.savefig(OUT / "figures/05_split_map.png", dpi=130)
    plt.close(fig)

    top = c.head(30).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(top.ecosys.str.slice(0, 55), top.n_train, color="#1565c0", label="train")
    ax.barh(top.ecosys.str.slice(0, 55), top.n_test, left=top.n_train, color="#c62828", label="test")
    ax.set_title(f"Final evaluation set: {len(d)} plots, {len(c)} Ecological Systems (top 30 shown)", fontsize=9)
    ax.set_xlabel("plots")
    ax.tick_params(axis="y", labelsize=7)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "figures/06_final_counts_by_split.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
