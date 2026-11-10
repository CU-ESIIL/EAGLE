"""Stage 3: write the evaluation tables with SatBird's own train/valid/test split (no new split is made).

Usage: `03_export.py [USA_winter|USA_summer]` (default USA_winter). Output folder datasets/SATBIRD_<season>_eval/, e.g.
SATBIRD_USA_winter_eval.

Rows: the hotspots of stage 2 with lidar collected inside the label years. Split: the `split` column of SatBird
(`train`, `val` = SatBird's `valid`, `test`), plus bool `test_split` (True = SatBird test; val is False, as in VegBench).
SatBird built its split by clustering hotspots with DBSCAN (5 km) and assigning clusters at random (70/15/15).

Targets: SatBird's encounter rates (share of complete checklists of the season reporting the species), the `probs` vector,
named by the scientific names in species_list.txt. Occupancy = encounter rate > 0. Per the many-classes rule every species
present at >= 1 hotspot in each of train and test is published once (min counts are an evaluation setting):
  location_encounter_rate.parquet   columns enc__<Genus_species>
  location_occupancy.parquet        columns occ__<Genus_species>  (int8)
  location_species_summary.csv      every species with its train/val/test counts and `included`
Counts use the final rows. Also writes catalog/species_split_counts_<season>.csv and figures/.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "datasets/raw/satbird"
CAT = Path(__file__).parent / "catalog"
MAX_YEAR_DIFF = 0  # lidar collected inside the label years (gap 0, see stage 2)
META = ["hotspot_id", "lat", "lon", "year", "state_code", "county_code", "num_complete_checklists", "num_different_species",
        "product_name_AWS", "collection_year_AWS", "year_diff_AWS", "label_year_min", "label_year_max", "als_site_id",
        "split", "test_split"]


def main(season):
    out = ROOT / f"datasets/SATBIRD_{season}_eval"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    src = pd.read_parquet(RAW / f"SatBird-labels/{season}/hotspots.parquet", columns=["hotspot_id", "probs"])
    species = (RAW / f"SatBird-labels/{season}/species_list.txt").read_text().splitlines()
    species = [s.strip().replace(" ", "_") for s in species]
    d = pd.read_parquet(RAW / f"derived/{season}_als_covered.parquet")
    d = d[d.year_diff_AWS <= MAX_YEAR_DIFF]
    d = d.merge(src, on="hotspot_id", how="left").reset_index(drop=True)
    rate = np.stack(d.probs.to_numpy()).astype("float32")
    assert rate.shape[1] == len(species)
    d["split"] = d.split.replace({"valid": "val"})
    d["test_split"] = d.split == "test"
    d = d[META]
    occ = rate > 0
    tr, va, te = ((d.split == s).to_numpy() for s in ("train", "val", "test"))

    summ = pd.DataFrame({
        "species": species,
        "n_hotspots_train": occ[tr].sum(0), "n_hotspots_val": occ[va].sum(0), "n_hotspots_test": occ[te].sum(0),
        "prevalence_train": occ[tr].mean(0).round(4), "prevalence_test": occ[te].mean(0).round(4),
        "mean_rate_train": rate[tr].mean(0).round(5), "mean_rate_test": rate[te].mean(0).round(5),
    })
    summ["included"] = (summ.n_hotspots_train >= 1) & (summ.n_hotspots_test >= 1)
    summ.to_csv(out / "location_species_summary.csv", index=False)
    summ.to_csv(CAT / f"species_split_counts_{season}.csv", index=False)
    idx = np.flatnonzero(summ.included)
    names = np.array(species)[idx]
    pd.concat([d, pd.DataFrame(rate[:, idx], columns=[f"enc__{x}" for x in names])], axis=1).to_parquet(
        out / "location_encounter_rate.parquet", index=False)
    pd.concat([d, pd.DataFrame(occ[:, idx].astype("int8"), columns=[f"occ__{x}" for x in names])], axis=1).to_parquet(
        out / "location_occupancy.parquet", index=False)
    print(f"{out.name}: {len(d)} hotspots (train {tr.sum()}, val {va.sum()}, test {te.sum()}), {len(idx)} of {len(species)} species; "
          f">=20 in train and test: {int(((summ.n_hotspots_train >= 20) & (summ.n_hotspots_test >= 20)).sum())}")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for sp, c in [("train", "tab:blue"), ("val", "tab:green"), ("test", "tab:orange")]:
        g = d[(d.split == sp) & d.lon.between(-126, -66) & d.lat.between(24, 50)]
        ax.scatter(g.lon, g.lat, s=2, alpha=0.5, color=c, label=f"{sp} ({int((d.split == sp).sum())} hotspots)")
    ax.set_title(f"SatBird {season} hotspots with lidar collected 2010-2023: SatBird's own split (CONUS shown)", fontsize=9)
    ax.legend(markerscale=5)
    fig.savefig(out / "figures/01_map_split.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    m = np.minimum(summ.n_hotspots_train, summ.n_hotspots_test)
    ax.hist(np.log10(m.clip(lower=0.5)), bins=40)
    for t in (1, 20):
        ax.axvline(np.log10(t), color="r", ls="--")
        ax.text(np.log10(t), ax.get_ylim()[1] * 0.9, f" >= {t}: {int((m >= t).sum())} sp", color="r", fontsize=8)
    ax.set_xlabel("log10( min(hotspots present in train, in test) )")
    ax.set_ylabel("species")
    fig.savefig(out / "figures/02_species_hotspots_per_split.png", dpi=130, bbox_inches="tight")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "USA_winter")
