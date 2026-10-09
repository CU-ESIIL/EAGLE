"""Stage 4: per-location species targets, one row per camera location with one matched ALS product.

Stage 3 gives one row per deployment. Here the deployments of the same site (`location_id`: coordinates
rounded to 4 decimals, ~11 m) are pooled:
  - each location gets ONE 3DEP product: the one matched by most of its deployments, ties broken by the smaller mean
    year gap, then by name. Only deployments matched to that product are pooled, so a label always refers to the
    lidar of its row (12 of 3,409 locations had deployments on two products; the other deployments are left out).
  - occupancy: 1 if the species was detected in any pooled deployment, else 0
  - detection rate: independent events (30-min rule) summed over the pooled deployments / total survey nights x 100

Species included: wild species (not human or domestic) with at least `--min-locations` N locations of presence in
each of the train and the test split (default 1 = at least one location in both). N = 1 writes to
datasets/SNAPSHOT_USA_eval/; any other N writes the same locations and file names to datasets/SNAPSHOT_USA_eval-N/
(`--min-locations 20` is the "-20" evaluation set).
Two separate tables are written, one for each target type; both carry the same location columns.

Reads datasets/SNAPSHOT_USA_eval/{snapshot_usa_eval.parquet, species_events_long.parquet, species_summary.csv} (stage 3).
Writes {location_occupancy.parquet, location_detection_rate.parquet, location_species_summary.csv} to the output folder.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from eagle_als.sites import add_site_ids  # noqa: E402

SRC = ROOT / "datasets/SNAPSHOT_USA_eval"


def mode(s):
    s = s.dropna()
    return s.mode().iat[0] if len(s) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-locations", type=int, default=1, help="minimum locations with the species present in each split")
    a = ap.parse_args()
    MIN_LOCATIONS_PER_SPLIT = a.min_locations
    OUT = SRC if a.min_locations == 1 else ROOT / f"datasets/SNAPSHOT_USA_eval-{a.min_locations}"
    OUT.mkdir(exist_ok=True)
    d = pd.read_parquet(SRC / "snapshot_usa_eval.parquet")
    d = d[[c for c in d.columns if not c.startswith(("occ__", "rate__"))]]
    ev = pd.read_parquet(SRC / "species_events_long.parquet")
    taxa = pd.read_csv(SRC / "species_summary.csv")[["species", "common_name", "taxon_class", "is_human_or_domestic"]]

    # one product per location
    pc = d.groupby(["location_id", "product_name_AWS"]).agg(n=("deployment_key", "size"), gap=("year_diff_AWS", "mean")).reset_index()
    best = pc.sort_values(["location_id", "n", "gap", "product_name_AWS"], ascending=[True, False, True, True]).drop_duplicates("location_id")
    keep = d.merge(best[["location_id", "product_name_AWS"]], on=["location_id", "product_name_AWS"])
    print(f"deployments: {len(d)} -> {len(keep)} pooled ({len(d) - len(keep)} on a second product at {int((pc.groupby('location_id').size() > 1).sum())} locations)")

    g = keep.groupby("location_id")
    loc = pd.DataFrame({
        "lat": g.lat.mean(), "lon": g.lon.mean(),
        "n_deployments": g.size(), "total_survey_nights": g.Survey_Nights.sum(),
        "first_year": g.year.min(), "last_year": g.year.max(),
        "Habitat": g.Habitat.agg(mode), "Development_Level": g.Development_Level.agg(mode),
        "Project": g.Project.agg(mode), "Camera_Trap_Array": g.Camera_Trap_Array.agg(mode),
        "n_species_wild": np.nan,
        "product_name_AWS": g.product_name_AWS.first(), "collection_year_AWS": g.collection_year_AWS.first(),
        "year_diff_AWS_mean": g.year_diff_AWS.mean().round(2), "year_diff_AWS_max": g.year_diff_AWS.max(),
        "test_split": g.test_split.first(),
    }).reset_index()
    assert (g.test_split.nunique() == 1).all()
    loc = add_site_ids(loc, "lat", "lon")

    # pooled species events
    e = ev.merge(keep[["deployment_key", "location_id"]], on="deployment_key").merge(taxa, on="species")
    pooled = e.groupby(["location_id", "species"]).agg(n_events=("n_events", "sum"), n_sequences=("n_sequences", "sum")).reset_index()
    pooled = pooled.merge(loc[["location_id", "total_survey_nights", "test_split"]], on="location_id").merge(taxa, on="species")
    pooled["rate"] = 100 * pooled.n_events / pooled.total_survey_nights
    wild = pooled[~pooled.is_human_or_domestic]
    loc["n_species_wild"] = loc.location_id.map(wild.groupby("location_id").species.nunique()).fillna(0).astype(int)

    # species included
    n_loc = {False: int((~loc.test_split).sum()), True: int(loc.test_split.sum())}
    s = pooled.groupby("species").agg(common_name=("common_name", "first"), taxon_class=("taxon_class", "first"),
                                      is_human_or_domestic=("is_human_or_domestic", "first"),
                                      n_locations=("location_id", "nunique"), n_events=("n_events", "sum")).reset_index()
    for flag, name in [(False, "train"), (True, "test")]:
        p = pooled[pooled.test_split == flag].groupby("species").location_id.nunique().rename(f"n_locations_{name}")
        s = s.merge(p, on="species", how="left")
        s[f"n_locations_{name}"] = s[f"n_locations_{name}"].fillna(0).astype(int)
        s[f"prevalence_{name}"] = (s[f"n_locations_{name}"] / n_loc[flag]).round(4)
    s["included"] = (~s.is_human_or_domestic) & (s.n_locations_train >= MIN_LOCATIONS_PER_SPLIT) & (s.n_locations_test >= MIN_LOCATIONS_PER_SPLIT)
    s = s.sort_values("n_locations", ascending=False).reset_index(drop=True)
    s.to_csv(OUT / "location_species_summary.csv", index=False)
    sp = s[s.included].species.tolist()

    col = lambda x: x.replace(" ", "_")
    w = wild[wild.species.isin(sp)]
    lead = [c for c in loc.columns]
    ids = loc.location_id
    occ = pd.DataFrame(0, index=ids, columns=[f"occ__{col(x)}" for x in sp], dtype="int8")
    rate = pd.DataFrame(0.0, index=ids, columns=[f"rate__{col(x)}" for x in sp])
    for r in w.itertuples(index=False):
        occ.at[r.location_id, f"occ__{col(r.species)}"] = 1
        rate.at[r.location_id, f"rate__{col(r.species)}"] = r.rate
    loc = loc[lead].sort_values("location_id").reset_index(drop=True)
    occ_t = pd.concat([loc, occ.reindex(loc.location_id).reset_index(drop=True)], axis=1)
    rate_t = pd.concat([loc, rate.reindex(loc.location_id).reset_index(drop=True)], axis=1)
    occ_t.to_parquet(OUT / "location_occupancy.parquet", index=False)
    rate_t.to_parquet(OUT / "location_detection_rate.parquet", index=False)

    print(f"{len(loc)} locations, {loc.als_site_id.nunique()} unique als_site_id, {loc.product_name_AWS.nunique()} products")
    print("test_split:", loc.test_split.value_counts().to_dict(), "; deployments per location:", loc.n_deployments.value_counts().sort_index().to_dict())
    print(f"species detected {len(s)}; wild {int((~s.is_human_or_domestic).sum())}; included (>= {MIN_LOCATIONS_PER_SPLIT} location in both splits): {len(sp)}")
    print("included species by locations present (train+test): ", s[s.included].n_locations.describe().round(1).to_dict())
    print("included with < 5 locations in either split:", int(((s.included) & ((s.n_locations_train < 5) | (s.n_locations_test < 5))).sum()),
          "; < 20:", int(((s.included) & ((s.n_locations_train < 20) | (s.n_locations_test < 20))).sum()))
    print("wild species dropped for missing from one split:", int(((~s.is_human_or_domestic) & ~s.included).sum()))


if __name__ == "__main__":
    main()
