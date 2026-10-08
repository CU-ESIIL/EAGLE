"""Build the NLCD land-cover evaluation table for lidar representations.

Source: NLCD 2021 accuracy-assessment (AA) reference points (Wickham et al. 2026,
https://doi.org/10.5066/P9JZ7AO3): 3,245 CONUS 30 m pixels, each interpreted by trained
analysts (blind to the map) for 2016, 2019 and 2021, with primary + alternate labels.
These are reference labels, not modeled NLCD.

Each point is matched to the 3DEP product whose collection year is closest to a label
year, using the same columns as the other datasets (product_name_AWS,
collection_year_AWS, year_diff_AWS). Only exact-year matches are kept (MAX_YEAR_DIFF). Footprint matching only; see 02_check_streaming.py
for the optional point-cloud availability check.

Raw inputs: datasets/raw/NLCD_AA2021/ (git-ignored). Output: datasets/NLCD_eval/.
"""

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from eagle_als.sites import add_site_ids  # noqa: E402

RAW = ROOT / "datasets/raw/NLCD_AA2021"
OUT = ROOT / "datasets/NLCD_eval"
MAX_YEAR_DIFF = 0  # exact year match only; widen to 1 if the set gets too small (<1k rows)
PER_CLASS = 100  # target size of the class-balanced subset
MAX_PER_PRODUCT = 10  # per class, so one 3DEP project cannot dominate
SEED = 0

L2 = {
    11: "Open Water", 12: "Perennial Ice/Snow",
    21: "Developed, Open Space", 22: "Developed, Low Intensity",
    23: "Developed, Medium Intensity", 24: "Developed, High Intensity",
    31: "Barren Land", 41: "Deciduous Forest", 42: "Evergreen Forest", 43: "Mixed Forest",
    52: "Shrub/Scrub", 71: "Grassland/Herbaceous", 81: "Pasture/Hay", 82: "Cultivated Crops",
    90: "Woody Wetlands", 95: "Emergent Herbaceous Wetlands",
}
L1 = {
    10: "Water", 20: "Developed", 30: "Barren", 40: "Forest", 50: "Shrubland",
    70: "Herbaceous", 80: "Planted/Cultivated", 90: "Wetlands",
}
LABEL_YEARS = {16: 2016, 19: 2019, 21: 2021}


def load_reference():
    x = pd.read_excel(RAW / "AA2021Fsort_LG_DGS_merge.xlsx")
    crs = gpd.read_file(RAW / "AA2021F.shp").crs
    pts = gpd.GeoSeries(gpd.points_from_xy(x.POINT_X, x.POINT_Y), crs=crs).to_crs(4326)
    x["lon"], x["lat"] = pts.x, pts.y
    # long format: one row per (point, label year)
    rows = []
    for yy, nominal in LABEL_YEARS.items():
        d = pd.DataFrame({
            "point_id": x.FID.astype(int),
            "lat": x.lat, "lon": x.lon,
            "label_year": nominal,
            "nlcd_code": pd.to_numeric(x[f"Lpri{yy}L2"], errors="coerce"),
            "alt_nlcd_code": pd.to_numeric(x[f"Lalt{yy}L2"], errors="coerce"),
            "label_image_year": pd.to_numeric(x[f"yr{yy}"], errors="coerce"),
            "label_conf": x["Conf"],
        })
        rows.append(d)
    d = pd.concat(rows, ignore_index=True)
    d = d[d.nlcd_code.isin(L2)].copy()
    d["nlcd_code"] = d.nlcd_code.astype(int)
    d["alt_nlcd_code"] = d.alt_nlcd_code.where(d.alt_nlcd_code.isin(L2)).astype("Int64")
    # observation year: image year if it is plausibly the map epoch, else nominal year
    ok = (d.label_image_year - d.label_year).abs() <= 2
    d["year"] = np.where(ok, d.label_image_year, d.label_year).astype(int)
    # stable = same primary label in all three epochs (safe to pair with any nearby ALS year)
    codes = d.pivot(index="point_id", columns="label_year", values="nlcd_code")
    stable = codes.nunique(axis=1).eq(1) & codes.notna().all(axis=1)
    d["stable_2016_2021"] = d.point_id.map(stable)
    return d


def match_als(d):
    res = gpd.read_file(ROOT / "datasets/USGS_3dep/usgs_3dep_resources_AWS.geojson")[
        ["name", "collection_year", "geometry"]
    ]
    pts = gpd.GeoDataFrame(
        d[["point_id"]].drop_duplicates().reset_index(drop=True),
        geometry=gpd.points_from_xy(
            d.drop_duplicates("point_id").lon, d.drop_duplicates("point_id").lat
        ),
        crs=4326,
    )
    j = gpd.sjoin(pts, res, predicate="within").drop(columns=["geometry", "index_right"])
    m = d.merge(j, on="point_id")
    m = m[m.collection_year.notna()]  # footprints without a collection year cannot be matched
    m["year_diff_AWS"] = (m.collection_year - m.year).abs().astype(int)
    m = m.rename(columns={"name": "product_name_AWS", "collection_year": "collection_year_AWS"})
    m["collection_year_AWS"] = m.collection_year_AWS.astype(int)
    # best (label year, product) per point: smallest gap, then stable-first, then latest label
    m = m.sort_values(["point_id", "year_diff_AWS", "label_year"], ascending=[True, True, False])
    m = m.drop_duplicates("point_id")
    return m[m.year_diff_AWS <= MAX_YEAR_DIFF].copy()


def balanced_subset(m):
    rng = np.random.default_rng(SEED)
    keep = []
    for _, g in m.groupby("nlcd_code"):
        g = g.assign(_r=rng.random(len(g)))
        # prefer small year gap and stable labels, then random; cap per product
        g = g.sort_values(["year_diff_AWS", "_r"])
        g = g.groupby("product_name_AWS").head(MAX_PER_PRODUCT)
        g = g.sort_values(["year_diff_AWS", "stable_2016_2021", "_r"], ascending=[True, False, True])
        keep.extend(g.head(PER_CLASS).point_id)
    return m.point_id.isin(keep)


def main():
    d = load_reference()
    m = match_als(d)
    m["nlcd_class"] = m.nlcd_code.map(L2)
    m["alt_nlcd_class"] = m.alt_nlcd_code.map(L2)
    m["nlcd_L1_code"] = (m.nlcd_code // 10 * 10).astype(int)
    m["nlcd_L1_class"] = m.nlcd_L1_code.map(L1)
    m["label_source"] = "NLCD2021_AA_reference"
    m = add_site_ids(m, "lat", "lon")
    # spatial split: ~25% of 1-degree blocks held out
    blk = (np.floor(m.lat).astype(int) * 1000 + np.floor(m.lon).astype(int)).astype(str)
    m["test_split"] = blk.map(lambda s: (hash_int(s) % 4) == 0)
    m["balanced_subset"] = balanced_subset(m)
    cols = [
        "point_id", "nlcd_code", "nlcd_class", "nlcd_L1_code", "nlcd_L1_class",
        "alt_nlcd_code", "alt_nlcd_class", "lat", "lon", "year", "label_year",
        "label_image_year", "label_conf", "stable_2016_2021", "label_source",
        "product_name_AWS", "collection_year_AWS", "year_diff_AWS", "als_site_id",
        "test_split", "balanced_subset",
    ]
    m = m[cols].sort_values(["nlcd_code", "point_id"]).reset_index(drop=True)
    OUT.mkdir(exist_ok=True)
    m.to_parquet(OUT / "nlcd_lidar_eval.parquet", index=False)
    print(f"{len(m)} points with ALS within {MAX_YEAR_DIFF} yr; {m.balanced_subset.sum()} in balanced subset")
    print(m.groupby(["nlcd_code", "nlcd_class"]).agg(
        n=("point_id", "size"), balanced=("balanced_subset", "sum"),
        stable=("stable_2016_2021", "sum"), exact_year=("year_diff_AWS", lambda s: (s == 0).sum()),
    ).to_string())


def hash_int(s):
    import zlib
    return zlib.crc32(s.encode())


if __name__ == "__main__":
    main()
