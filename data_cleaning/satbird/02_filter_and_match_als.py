"""Stage 2: match each SatBird hotspot to a 3DEP lidar (ALS) product collected inside the period the labels cover.

One row = one eBird hotspot of the original SatBird dataset (`hotspots.parquet` of sammlapp/SatBird-labels). Coordinates,
labels and the train/valid/test split are SatBird's own and are used as published: no hotspot is dropped or moved for any
reason other than missing lidar. Usage: `02_filter_and_match_als.py [USA_winter|USA_summer]` (default USA_winter).

The labels POOL the complete checklists of one season over 2010-2023 (Teng et al. 2023), so there is no single survey
year. A lidar product is therefore a valid match when its collection year lies anywhere in LABEL_YEARS = 2010..2023, not
within N years of one date. Among the products covering a point, the one with the smallest gap to the window is kept
(gap 0 = inside); ties go to the LATEST collection year. `year_diff_AWS` is this gap (0 = inside the window, else years
to the nearest window edge; the eval table keeps gap 0 only). `year` = REFERENCE_YEAR (2022, the Sentinel-2 year of the
original dataset) fills the standard column; it is not a survey date. Caveat: with a 14-year window the lidar can be
up to 13 years from some checklists.

Filters, in order (counts after each step go to catalog/filter_counts_<season>.csv):
  1. hotspots in SatBird (all of them)
  2. valid coordinates (inside lat/lon range, not 0/0)
  3. a 3DEP footprint covers the point with a collection year
  4. a covering product was collected inside LABEL_YEARS (gap 0)

Reads datasets/raw/satbird/SatBird-labels/<season>/hotspots.parquet (stage 1). Writes
datasets/raw/satbird/derived/<season>_als_covered.parquet (every covered hotspot with its gap, so a looser rule can be
applied later), catalog/filter_counts_<season>.csv and catalog/year_gap_counts_<season>.csv (rows kept for gap <= N, N = 0..5,
per SatBird split).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/streaming/3dep"))
from eagle_als.sites import add_site_ids  # noqa: E402
from get_als import lookup_3dep_by_latlon  # noqa: E402

RAW = ROOT / "datasets/raw/satbird"
DERIVED = RAW / "derived"
CAT = Path(__file__).parent / "catalog"
LABEL_YEARS = (2010, 2023)  # checklist years pooled in the SatBird labels
REFERENCE_YEAR = 2022
MAX_YEAR_DIFF = 0
MAX_REPORT = 5


def main(season):
    CAT.mkdir(exist_ok=True)
    DERIVED.mkdir(exist_ok=True)
    d = pd.read_parquet(RAW / f"SatBird-labels/{season}/hotspots.parquet", columns=[
        "hotspot_id", "lat", "lon", "state_code", "county_code", "num_complete_checklists", "num_different_species", "split"])
    counts = [("1. hotspots in SatBird", len(d))]
    d = d[d.lat.between(-90, 90) & d.lon.between(-180, 180) & ~((d.lat == 0) & (d.lon == 0))].copy()
    counts.append(("2. valid coordinates", len(d)))

    d["year"] = REFERENCE_YEAR
    keys = d[["lat", "lon"]].drop_duplicates()
    res = []
    for la, lo in keys.itertuples(index=False):
        m = lookup_3dep_by_latlon(float(la), float(lo), source="AWS")
        m = m[m.collection_year.notna()]
        if m.empty:
            res.append({})
            continue
        cy = m.collection_year.astype(int).to_numpy()
        gap = np.maximum(LABEL_YEARS[0] - cy, 0) + np.maximum(cy - LABEL_YEARS[1], 0)
        i = np.lexsort((-cy, gap))[0]  # smallest gap, then latest collection year
        res.append({"product_name_AWS": m.name.iloc[i], "collection_year_AWS": int(cy[i]), "year_diff_AWS": int(gap[i])})
    keys = pd.concat([keys.reset_index(drop=True), pd.DataFrame(res)], axis=1)
    d = d.merge(keys, on=["lat", "lon"], how="left")
    d = d[d.product_name_AWS.notna() & d.year_diff_AWS.notna()].copy()
    counts.append(("3. covered by a 3DEP footprint with a collection year", len(d)))
    d["label_year_min"], d["label_year_max"] = LABEL_YEARS
    d["collection_year_AWS"] = d.collection_year_AWS.astype(int)
    d["year_diff_AWS"] = d.year_diff_AWS.astype(int)
    d = add_site_ids(d, "lat", "lon")
    d.to_parquet(DERIVED / f"{season}_als_covered.parquet", index=False)

    gap = pd.DataFrame({"max_years_outside_window_N": range(MAX_REPORT + 1)})
    for s in ["train", "valid", "test"]:
        gap[f"n_{s}"] = [int(((d.year_diff_AWS <= n) & (d.split == s)).sum()) for n in gap.max_years_outside_window_N]
    gap["n_hotspots"] = gap[["n_train", "n_valid", "n_test"]].sum(axis=1)
    gap.to_csv(CAT / f"year_gap_counts_{season}.csv", index=False)
    d = d[d.year_diff_AWS <= MAX_YEAR_DIFF]
    counts.append((f"4. 3DEP collection year inside the label years {LABEL_YEARS[0]}-{LABEL_YEARS[1]}", len(d)))

    fc = pd.DataFrame(counts, columns=["step", "n_hotspots"])
    fc["dropped"] = (-fc.n_hotspots.diff()).fillna(0).astype(int)
    fc.to_csv(CAT / f"filter_counts_{season}.csv", index=False)
    print(fc.to_string(index=False))
    print(gap.to_string(index=False))
    print(f"{d.als_site_id.nunique()} unique als_site_id, {d.product_name_AWS.nunique()} lidar products")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "USA_winter")
