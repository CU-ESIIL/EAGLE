"""Stage 3: spatial train/test split and the four species-target variants.

Split: whole 1-degree lat/lon blocks go to test (~TEST_FRACTION of hotspots), blocks shuffled with SEED. SatBird's own
distance-based split is only in the 152 GB archive, so this split is our own and not comparable to SatBird's published scores.

Targets are SatBird's encounter rates: the share of complete June-July checklists (2010-2023) at the hotspot that
report the species. Occupancy = encounter rate > 0 (reported on at least one checklist). The species names are not
public outside the Drive archive, so columns are named by index in the 670-long target vector (`sp000` ... `sp669`);
`is_songbird` comes from SatBird's songbird_indices.npy.

Two species sets (the evaluation guide's "all" and ">= 20" variants), each with two target types, giving 4 tables:
  datasets/SATBIRD_eval/     species with >= 1 hotspot present in both train and test    ("all")
  datasets/SATBIRD_eval-20/  species with >= 20 hotspots present in both train and test  ("-20")
each holding location_encounter_rate.parquet (columns enc__spNNN), location_occupancy.parquet (occ__spNNN, int8),
location_species_summary.csv (every species, with `included`), and the shared hotspot columns.

Reads datasets/raw/satbird/derived/hotspots_als_covered.parquet (stage 2), the targets pickle and songbird_indices.npy.
Writes the folders above, catalog/species_split_counts.csv, and figures/ in datasets/SATBIRD_eval/.
"""
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "datasets/raw/satbird"
CAT = Path(__file__).parent / "catalog"
SEED = 20261029
TEST_FRACTION = 0.3
BLOCK_DEG = 1.0
MAX_YEAR_DIFF = 0  # lidar collected inside the label years 2010-2023 (gap 0, see stage 2)
MIN_VARIANTS = {1: "SATBIRD_eval", 20: "SATBIRD_eval-20"}
META = ["hotspot_id", "hotspot_name", "lat", "lon", "year", "state_code", "county_code", "n_checklists_all_time",
        "shift_m_vs_2023", "product_name_AWS", "collection_year_AWS", "year_diff_AWS", "label_year_min", "label_year_max", "als_site_id", "block", "test_split"]


def spatial_split(d):
    block = np.floor(d.lat / BLOCK_DEG).astype(int).astype(str) + "_" + np.floor(d.lon / BLOCK_DEG).astype(int).astype(str)
    sizes = block.value_counts()
    order = np.sort(sizes.index.to_numpy())
    np.random.default_rng(SEED).shuffle(order)
    target, n, test = TEST_FRACTION * len(d), 0, set()
    for b in order:
        if abs(n + sizes[b] - target) < abs(n - target):
            test.add(b)
            n += sizes[b]
    return block.isin(test), block


def main():
    CAT.mkdir(exist_ok=True)
    d = pd.read_parquet(RAW / "derived/hotspots_als_covered.parquet")
    d = d[d.year_diff_AWS <= MAX_YEAR_DIFF].reset_index(drop=True)
    targets = pickle.load(open(RAW / "satbird_usa_summer_targets.pkl", "rb"))
    rate = np.stack([targets[h] for h in d.hotspot_id]).astype("float32")
    n_sp = rate.shape[1]
    songbird = np.zeros(n_sp, bool)
    songbird[np.load(RAW / "songbird_indices.npy")] = True
    assert songbird.sum() + np.isin(np.arange(n_sp), np.load(RAW / "nonsongbird_indices.npy")).sum() == n_sp
    d["test_split"], d["block"] = spatial_split(d)
    d = d[META]
    occ = rate > 0
    te = d.test_split.to_numpy()

    summ = pd.DataFrame({
        "species": [f"sp{i:03d}" for i in range(n_sp)], "is_songbird": songbird,
        "n_hotspots_train": occ[~te].sum(0), "n_hotspots_test": occ[te].sum(0),
        "prevalence_train": occ[~te].mean(0).round(4), "prevalence_test": occ[te].mean(0).round(4),
        "mean_rate_train": rate[~te].mean(0).round(5), "mean_rate_test": rate[te].mean(0).round(5),
    })
    summ.to_csv(CAT / "species_split_counts.csv", index=False)

    for n_min, folder in MIN_VARIANTS.items():
        out = ROOT / "datasets" / folder
        out.mkdir(exist_ok=True)
        s = summ.copy()
        s["included"] = (s.n_hotspots_train >= n_min) & (s.n_hotspots_test >= n_min)
        s.to_csv(out / "location_species_summary.csv", index=False)
        idx = np.flatnonzero(s.included)
        names = s.species.to_numpy()[idx]
        enc = pd.concat([d, pd.DataFrame(rate[:, idx], columns=[f"enc__{x}" for x in names])], axis=1)
        occt = pd.concat([d, pd.DataFrame(occ[:, idx].astype("int8"), columns=[f"occ__{x}" for x in names])], axis=1)
        enc.to_parquet(out / "location_encounter_rate.parquet", index=False)
        occt.to_parquet(out / "location_occupancy.parquet", index=False)
        print(f"{folder}: {len(d)} hotspots ({int(te.sum())} test), {len(idx)} species "
              f"({int(songbird[idx].sum())} songbirds)")

    print("blocks:", d.block.nunique(), "; test fraction", round(te.mean(), 3))
    fig_dir = ROOT / "datasets/SATBIRD_eval/figures"
    fig_dir.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for flag, c, nm in [(False, "tab:blue", "train"), (True, "tab:orange", "test")]:
        g = d[(d.test_split == flag) & d.lon.between(-126, -66) & d.lat.between(24, 50)]
        ax.scatter(g.lon, g.lat, s=2, alpha=0.5, color=c, label=f"{nm} ({int((d.test_split == flag).sum())} hotspots)")
    ax.set_title("SatBird hotspots with lidar collected 2010-2023: 1-degree block split (CONUS shown)", fontsize=9)
    ax.legend(markerscale=5)
    fig.savefig(fig_dir / "01_map_split.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    m = np.minimum(summ.n_hotspots_train, summ.n_hotspots_test)
    ax.hist(np.log10(m.clip(lower=0.5)), bins=40)
    for t in (1, 20):
        ax.axvline(np.log10(t), color="r", ls="--")
        ax.text(np.log10(t), ax.get_ylim()[1] * 0.9, f" >= {t}: {int((m >= t).sum())} sp", color="r", fontsize=8)
    ax.set_xlabel("log10( min(hotspots present in train, in test) )")
    ax.set_ylabel("species")
    fig.savefig(fig_dir / "02_species_hotspots_per_split.png", dpi=130, bbox_inches="tight")


if __name__ == "__main__":
    main()
