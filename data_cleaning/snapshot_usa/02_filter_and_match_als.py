"""Stage 2: keep deployments with a precise location and enough survey effort, and match each to the
3DEP lidar (ALS) product closest in time.

Filters, in order (counts after each step go to catalog/filter_counts_stage2.csv):
  1. valid coordinates
  2. >= MIN_DECIMALS decimals in lat and lon in the source text (4 decimals ~ 11 m). Some deployments
     are rounded to 1-3 decimals (100 m - 10 km), too coarse for a 100 m-radius cookie
  3. >= MIN_NIGHTS survey nights (presence/absence of a species at a 1-6 night deployment says little)
  4. a 3DEP footprint covers the point (get_nearest_year_product, source = AWS)
  5. |collection year - survey year| <= MAX_YEAR_DIFF; the survey year is the year of Start_Date

`location_id` groups deployments of the same camera site across deployments and years (coordinates rounded
to 4 decimals, ~11 m); the same site is often re-surveyed with slightly different GPS fixes.

Reads datasets/raw/snapshotUSA_2019-2023/derived/deployments_clean.parquet (stage 1).
Writes datasets/raw/snapshotUSA_2019-2023/derived/deployments_matched.parquet, catalog/filter_counts_stage2.csv and
catalog/year_gap_distribution.csv (how many covered deployments fall at each year gap).
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/streaming/3dep"))
from eagle_als.sites import add_site_ids  # noqa: E402
from get_als import get_nearest_year_product  # noqa: E402

DERIVED = ROOT / "datasets/raw/snapshotUSA_2019-2023/derived"
CAT = Path(__file__).parent / "catalog"
MIN_DECIMALS = 4
MIN_NIGHTS = 7
MAX_YEAR_DIFF = 3


def main():
    CAT.mkdir(exist_ok=True)
    d = pd.read_parquet(DERIVED / "deployments_clean.parquet")
    counts = [("all deployments", len(d))]

    def keep(d, mask, label):
        d = d[mask.fillna(False)]
        counts.append((label, len(d)))
        return d

    d = d.rename(columns={"Latitude": "lat", "Longitude": "lon"})
    d["year"] = d.Start_Date.dt.year
    d = keep(d, d.lat.between(-90, 90) & d.lon.between(-180, 180) & ~((d.lat == 0) & (d.lon == 0)), "1. valid coordinates")
    d = keep(d, d.n_decimals >= MIN_DECIMALS, f"2. >= {MIN_DECIMALS} decimals in lat and lon")
    d = keep(d, d.Survey_Nights >= MIN_NIGHTS, f"3. >= {MIN_NIGHTS} survey nights")

    keys = d[["lat", "lon", "year"]].drop_duplicates()
    res = []
    for la, lo, yr in keys.itertuples(index=False):
        try:
            r = get_nearest_year_product(float(la), float(lo), int(yr), source="AWS")
        except Exception:
            r = None
        res.append(r or {})
    d = d.merge(pd.concat([keys.reset_index(drop=True), pd.DataFrame(res, index=range(len(keys)))], axis=1),
                on=["lat", "lon", "year"], how="left")
    d = keep(d, d.product_name_AWS.notna(), "4. covered by a 3DEP footprint with a collection year")
    gap = d.year_diff_AWS.astype(int).value_counts().sort_index().rename("n_deployments").rename_axis("year_diff_AWS").reset_index()
    gap.to_csv(CAT / "year_gap_distribution.csv", index=False)
    d = keep(d, d.year_diff_AWS <= MAX_YEAR_DIFF, f"5. 3DEP collection year within +/-{MAX_YEAR_DIFF} yr of the survey")

    d["collection_year_AWS"] = d.collection_year_AWS.astype(int)
    d["year_diff_AWS"] = d.year_diff_AWS.astype(int)
    d["location_id"] = d.lat.round(4).astype(str) + "_" + d.lon.round(4).astype(str)
    d = add_site_ids(d, "lat", "lon")
    d.to_parquet(DERIVED / "deployments_matched.parquet", index=False)

    fc = pd.DataFrame(counts, columns=["step", "n_deployments"])
    fc["dropped"] = (-fc.n_deployments.diff()).fillna(0).astype(int)
    fc.to_csv(CAT / "filter_counts_stage2.csv", index=False)
    print(fc.to_string(index=False))
    print("year gap among covered deployments:", dict(zip(gap.year_diff_AWS, gap.n_deployments)))
    print(f"matched: {len(d)} deployments, {d.location_id.nunique()} locations, {d.als_site_id.nunique()} unique als_site_id, "
          f"{d.product_name_AWS.nunique()} lidar products")
    print("by year:", d.year.value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
