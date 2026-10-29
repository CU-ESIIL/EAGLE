"""Match VegBench surveys (California vegetation plots, NVC labels) to the 3DEP lidar (ALS) product closest in time.

VegBench ships its own train/val/test split (`split`, `test_type`), which is kept unchanged.

Steps (counts go to catalog/):
  1. read benchmark_splits/VegBench/VegBench.csv (42,393 surveys; survey year from SurveyDate)
  2. a 3DEP footprint covers the point (src/streaming/3dep/get_als.get_nearest_year_product, source = AWS),
     which returns the covering product whose collection year is closest to the survey year
  3. report how many surveys remain for each maximum year gap N = 0..MAX_REPORT (catalog/year_gap_counts.csv,
     catalog/year_gap_counts_by_split.csv)
  4. keep |collection year - survey year| <= MAX_YEAR_DIFF (3, as in the other datasets) for the eval table

Outputs:
  datasets/raw/vegbench/derived/vegbench_als_covered.parquet  every covered survey with its year gap (choose any N)
  datasets/VegBench_eval/vegbench_eval.parquet                covered surveys with year_diff_AWS <= MAX_YEAR_DIFF
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/streaming/3dep"))
from eagle_als.sites import add_site_ids  # noqa: E402
from get_als import get_nearest_year_product  # noqa: E402

RAW = ROOT / "datasets/raw/vegbench"
CSV = RAW / "benchmark_splits/VegBench/VegBench.csv"
DERIVED = RAW / "derived"
OUT = ROOT / "datasets/VegBench_eval"
CAT = Path(__file__).parent / "catalog"
MAX_YEAR_DIFF = 3
MAX_REPORT = 5

KEEP = [
    "survey_point_id", "source_survey_id", "ProjectCode", "SurveyType", "SurveyDate", "year", "lat", "lon",
    "split", "test_type", "holdout_reason", "proximity_component_id",
    # NVC hierarchy, coarsest to finest
    "survey_formation_class", "survey_formation_subclass", "survey_formation", "survey_division",
    "survey_macrogroup", "survey_group", "survey_alliance", "survey_alliance_ca_code",
    "survey_association", "survey_association_ca_code",
    # crosswalks
    "survey_calveg_codes", "survey_calveg_names", "survey_cwhr_codes", "survey_cwhr_names",
    "survey_whr13_codes", "survey_whr13_names",
]


def main():
    DERIVED.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
    CAT.mkdir(exist_ok=True)
    d = pd.read_csv(CSV, low_memory=False)
    d = d.rename(columns={"latitude": "lat", "longitude": "lon"})
    d["year"] = pd.to_datetime(d.SurveyDate, utc=True).dt.year.astype(int)
    d["split"] = d.split.astype(str)

    keys = d[["lat", "lon", "year"]].drop_duplicates()
    res = []
    for la, lo, yr in keys.itertuples(index=False):
        try:
            r = get_nearest_year_product(float(la), float(lo), int(yr), source="AWS")
        except Exception:  # footprint without a collection year
            r = None
        res.append(r or {})
    hits = pd.concat([keys.reset_index(drop=True), pd.DataFrame(res, index=range(len(keys)))], axis=1)
    d = d.merge(hits, on=["lat", "lon", "year"], how="left")

    cov = d[d.product_name_AWS.notna()].copy()
    cov["collection_year_AWS"] = cov.collection_year_AWS.astype(int)
    cov["year_diff_AWS"] = cov.year_diff_AWS.astype(int)
    cov["signed_year_diff"] = cov.collection_year_AWS - cov.year
    cov = add_site_ids(cov, "lat", "lon")
    cols = KEEP + ["product_name_AWS", "collection_year_AWS", "year_diff_AWS", "signed_year_diff", "als_site_id"]
    cov = cov[cols]
    cov.to_parquet(DERIVED / "vegbench_als_covered.parquet", index=False)

    d["group"] = d.split + d.test_type.fillna("").map(lambda t: f"/{t}" if t else "")
    cov["group"] = d.loc[cov.index, "group"]
    rows = [("all surveys", len(d)), ("covered by a 3DEP footprint with a collection year", len(cov))]
    for n in range(MAX_REPORT + 1):
        rows.append((f"|collection year - survey year| <= {n}", int((cov.year_diff_AWS <= n).sum())))
    fc = pd.DataFrame(rows, columns=["step", "n_surveys"])
    fc.to_csv(CAT / "year_gap_counts.csv", index=False)

    by = {"all": d.group.value_counts(), "covered": cov.group.value_counts()}
    for n in range(MAX_REPORT + 1):
        by[f"N={n}"] = cov[cov.year_diff_AWS <= n].group.value_counts()
    by = pd.DataFrame(by).fillna(0).astype(int)
    by.to_csv(CAT / "year_gap_counts_by_split.csv")

    gap = cov.year_diff_AWS.value_counts().sort_index().rename("n_surveys").rename_axis("year_diff_AWS")
    gap.to_csv(CAT / "year_gap_distribution.csv")

    ev = cov[cov.year_diff_AWS <= MAX_YEAR_DIFF].drop(columns="group")
    ev.to_parquet(OUT / "vegbench_eval.parquet", index=False)

    print(fc.to_string(index=False))
    print(by.to_string())
    print("year gap among covered:", gap.to_dict())
    print(f"eval table: {len(ev)} surveys, {ev.als_site_id.nunique()} unique als_site_id, "
          f"{ev.product_name_AWS.nunique()} lidar products, {ev.survey_alliance.nunique()} alliances")


if __name__ == "__main__":
    main()
