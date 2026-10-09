"""Stage 3: build the species occupancy / encounter-rate targets, make the 50% spatial test hold-out and export.

Unit of the table: one camera deployment (one camera at one site for one survey). A species that was not
detected in a deployment is an absence (0 events), with the survey effort in `Survey_Nights`.

Targets, per species (names use the scientific name with `_` for spaces):
  occ__<species>   presence (1) / absence (0): at least one sequence in the deployment
  rate__<species>  encounter rate: independent events (30-min rule, stage 1) per 100 camera-nights
Wide columns are written only for ELIGIBLE species: wild species (not human or domestic) detected in at least
MIN_PRESENT_PER_SPLIT deployments of both the train and the test split. Every species is in
the long table `species_events_long.parquet` and in `species_summary.csv`.

Split: ~50% of deployments held out, assigned by whole 0.1-degree lat/lon blocks (spatial hold-out; cameras
of one array and repeat surveys of one site share blocks, so a random split would leak). Blocks are shuffled
with SEED (same seed as the BBS, butterfly and LFRDB splits) and added to the test set while that moves the test
share closer to 50%.

Reads datasets/raw/snapshotUSA_2019-2023/derived/{deployments_matched,species_counts,unresolved_counts,taxa}.parquet.
Writes datasets/SNAPSHOT_USA_eval/{snapshot_usa_eval.parquet, species_events_long.parquet, species_summary.csv, figures/}.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DERIVED = ROOT / "datasets/raw/snapshotUSA_2019-2023/derived"
OUT = ROOT / "datasets/SNAPSHOT_USA_eval"
SEED = 20260909
TEST_FRACTION = 0.5
MIN_PRESENT_PER_SPLIT = 20


def spatial_split(d):
    block = np.floor(d.lat * 10).astype(int).astype(str) + "_" + np.floor(d.lon * 10).astype(int).astype(str)
    sizes = block.value_counts()
    order = np.sort(sizes.index.to_numpy())  # sorted, so the shuffle depends only on SEED
    np.random.default_rng(SEED).shuffle(order)
    target = TEST_FRACTION * len(d)
    test, n = set(), 0
    for b in order:
        if abs(n + sizes[b] - target) < abs(n - target):
            test.add(b)
            n += sizes[b]
    return block.isin(test), block


def main():
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    d = pd.read_parquet(DERIVED / "deployments_matched.parquet")
    counts = pd.read_parquet(DERIVED / "species_counts.parquet")
    unres = pd.read_parquet(DERIVED / "unresolved_counts.parquet")
    taxa = pd.read_parquet(DERIVED / "taxa.parquet")

    d = d.rename(columns={"Year": "ssusa_year", "Deployment_ID": "deployment_id"})
    d["deployment_key"] = d.ssusa_year.astype(str) + "|" + d.deployment_id
    d["Feature_Type"] = d.Feature_Type.where(d.Feature_Type.fillna("").str.strip() != "")
    d["test_split"], block = spatial_split(d)

    counts = counts.rename(columns={"Year": "ssusa_year", "Deployment_ID": "deployment_id"})
    counts["deployment_key"] = counts.ssusa_year.astype(str) + "|" + counts.deployment_id
    counts = counts[counts.deployment_key.isin(d.deployment_key)].merge(
        d[["deployment_key", "Survey_Nights", "test_split"]], on="deployment_key").merge(
        taxa[["species", "common_name", "taxon_class", "is_human_or_domestic"]], on="species")
    counts["encounter_rate_per_100n"] = 100 * counts.n_events / counts.Survey_Nights

    # species summary and eligibility
    n_dep = {False: (~d.test_split).sum(), True: d.test_split.sum()}
    g = counts.groupby("species")
    summ = g.agg(common_name=("common_name", "first"), taxon_class=("taxon_class", "first"),
                 is_human_or_domestic=("is_human_or_domestic", "first"), n_sequences=("n_sequences", "sum"),
                 n_events=("n_events", "sum"), n_deployments=("deployment_key", "nunique")).reset_index()
    for flag, name in [(False, "train"), (True, "test")]:
        p = counts[counts.test_split == flag].groupby("species").deployment_key.nunique().rename(f"n_present_{name}")
        summ = summ.merge(p, on="species", how="left")
        summ[f"n_present_{name}"] = summ[f"n_present_{name}"].fillna(0).astype(int)
        summ[f"prevalence_{name}"] = (summ[f"n_present_{name}"] / n_dep[flag]).round(4)
    summ["n_locations"] = summ.species.map(counts.merge(d[["deployment_key", "location_id"]], on="deployment_key")
                                           .groupby("species").location_id.nunique())
    summ["eligible"] = (~summ.is_human_or_domestic) & (summ.n_present_train >= MIN_PRESENT_PER_SPLIT) & (summ.n_present_test >= MIN_PRESENT_PER_SPLIT)
    summ = summ.sort_values("n_deployments", ascending=False).reset_index(drop=True)
    summ.to_csv(OUT / "species_summary.csv", index=False)
    elig = summ[summ.eligible].species.tolist()

    # wide targets for eligible species
    sub = counts[counts.species.isin(elig)]
    occ = sub.pivot(index="deployment_key", columns="species", values="n_events").reindex(d.deployment_key).reindex(columns=elig)
    present = occ.notna().astype(int)
    rate = (occ.fillna(0).to_numpy() * 100 / d.Survey_Nights.to_numpy()[:, None])
    col = lambda s: s.replace(" ", "_")
    wide = pd.concat([present.set_axis([f"occ__{col(s)}" for s in elig], axis=1).reset_index(drop=True),
                      pd.DataFrame(rate, columns=[f"rate__{col(s)}" for s in elig])], axis=1)

    wild = counts[~counts.is_human_or_domestic].groupby("deployment_key").species.nunique().rename("n_species_wild")
    d = d.merge(wild, left_on="deployment_key", right_index=True, how="left")
    d["n_species_wild"] = d.n_species_wild.fillna(0).astype(int)
    d = d.merge(unres.rename(columns={"Year": "ssusa_year", "Deployment_ID": "deployment_id"}), on=["ssusa_year", "deployment_id"], how="left")
    d["n_seq_unresolved"] = d.n_seq_unresolved.fillna(0).astype(int)

    lead = ["deployment_key", "deployment_id", "ssusa_year", "Project", "Camera_Trap_Array", "Site_Name", "location_id",
            "lat", "lon", "n_decimals", "year", "Start_Date", "End_Date", "Survey_Nights",
            "Habitat", "Development_Level", "Feature_Type", "n_species_wild", "n_seq_unresolved",
            "product_name_AWS", "collection_year_AWS", "year_diff_AWS", "als_site_id", "test_split"]
    out = pd.concat([d[lead].reset_index(drop=True), wide], axis=1).sort_values("deployment_key").reset_index(drop=True)
    out.to_parquet(OUT / "snapshot_usa_eval.parquet", index=False)
    counts[["deployment_key", "species", "common_name", "n_sequences", "n_events", "encounter_rate_per_100n"]] \
        .sort_values(["deployment_key", "species"]).to_parquet(OUT / "species_events_long.parquet", index=False)

    print(f"{len(out)} deployments, {out.location_id.nunique()} locations, {block.nunique()} 0.1-degree blocks, {len(elig)} eligible species")
    print("test_split:", out.test_split.value_counts().to_dict(), f"({out.test_split.mean():.3f} test)")
    print("locations in both splits (should be 0):", int((out.groupby("location_id").test_split.nunique() > 1).sum()))
    print("species detected:", len(summ), "; wild:", int((~summ.is_human_or_domestic).sum()),
          f"; eligible (>= {MIN_PRESENT_PER_SPLIT} deployments in both splits): {len(elig)}")
    print("Habitat:", out.Habitat.value_counts().to_dict())
    print("Development_Level:", out.Development_Level.value_counts().to_dict())
    print(summ[summ.eligible][["species", "common_name", "n_present_train", "n_present_test", "prevalence_train", "prevalence_test"]].to_string(index=False))

    # figures
    fig, ax = plt.subplots(figsize=(8, 5))
    for flag, color, name in [(False, "#1565c0", "train"), (True, "#c62828", "test")]:
        g_ = out[(out.test_split == flag) & out.lon.between(-126, -66) & out.lat.between(24, 50)]
        ax.scatter(g_.lon, g_.lat, s=4, alpha=0.5, color=color, label=f"{name} ({int((out.test_split == flag).sum())} deployments)")
    ax.set_title("Snapshot USA deployments with lidar within 3 yr: train/test by 0.1-degree block (CONUS shown)", fontsize=9)
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    ax.legend(fontsize=8, markerscale=2)
    fig.tight_layout()
    fig.savefig(OUT / "figures/01_split_map.png", dpi=130)
    plt.close(fig)

    e = summ[summ.eligible].sort_values("n_deployments").tail(40)
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.barh(e.common_name.str.slice(0, 28), e.n_present_train, color="#1565c0", label="train")
    ax.barh(e.common_name.str.slice(0, 28), e.n_present_test, left=e.n_present_train, color="#c62828", label="test")
    ax.set_title(f"Deployments with the species present, by split ({len(elig)} eligible species; top 40 shown)", fontsize=9)
    ax.set_xlabel(f"deployments (of {len(out)})")
    ax.tick_params(axis="y", labelsize=7)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "figures/02_species_presence_by_split.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
