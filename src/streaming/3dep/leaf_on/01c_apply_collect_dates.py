"""Step 1c: reconcile WESM dates (step 1) with point-cloud GpsTime (step 1b) and write them to the registry.

Policy per product:
  WESM dates present, GpsTime median within +-45 d of the WESM window -> WESM dates, check='confirmed_by_gpstime'
  WESM dates present, no usable GpsTime                            -> WESM dates, check='gpstime_unavailable'
  WESM dates present, GpsTime disagrees                            -> WESM dates, check='conflict_wesm_vs_gpstime'
                                                                      (inspect; gps_* columns in gpstime_dates.csv)
  no WESM dates, usable GpsTime                                    -> GpsTime p01..p99 span, check='gpstime_only'
  neither                                                          -> null, check='no_date'
collect_dates_source records where the WESM dates came from (exact workunit name vs normalized match).

Registry changes (datasets/USGS_3dep/usgs_3dep_resources_AWS.geojson): collect_start / collect_end are
replaced by the reconciled dates, collection_year is recomputed as the year of collect_start, the old value is kept
in collection_year_prev, and collect_dates_source / collect_dates_check are added.
Also writes datasets/USGS_3dep/leaf_on/collect_dates_final.csv (used by steps 3 and 4) and
collection_year_changes.csv (every product whose collection_year changed).
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[4]
D = REPO / "datasets/USGS_3dep/leaf_on"
AWS = REPO / "datasets/USGS_3dep/usgs_3dep_resources_AWS.geojson"
PAD = pd.Timedelta(days=45)

aws = gpd.read_file(AWS)
if "collection_year_prev" not in aws.columns:  # first run: remember the original value
    aws["collection_year_prev"] = aws["collection_year"]

w = pd.read_csv(D / "collect_dates.csv", parse_dates=["collect_start", "collect_end"])
g = pd.read_csv(D / "gpstime_dates.csv", parse_dates=["gps_p01", "gps_median", "gps_p99"])
f = w.merge(g[["name", "gps_p01", "gps_median", "gps_p99", "gps_status"]], on="name", how="left")

has_w = f["collect_start"].notna()
has_g = f["gps_status"].eq("ok")
inside = (f["gps_median"] >= f["collect_start"] - PAD) & (f["gps_median"] <= f["collect_end"] + PAD)
f["dates_check"] = np.select(
    [has_w & has_g & inside, has_w & has_g & ~inside, has_w & ~has_g, ~has_w & has_g],
    ["confirmed_by_gpstime", "conflict_wesm_vs_gpstime", "gpstime_unavailable", "gpstime_only"],
    "no_date",
)
gps_only = f["dates_check"].eq("gpstime_only")
f.loc[gps_only, "collect_start"] = f.loc[gps_only, "gps_p01"]
f.loc[gps_only, "collect_end"] = f.loc[gps_only, "gps_p99"]
f.loc[gps_only, "dates_source"] = "gpstime"
f.to_csv(D / "collect_dates_final.csv", index=False, date_format="%Y-%m-%d")

fi = f.set_index("name").reindex(aws["name"])
aws["collect_start"] = pd.to_datetime(fi["collect_start"].values)
aws["collect_end"] = pd.to_datetime(fi["collect_end"].values)
aws["collect_dates_source"] = fi["dates_source"].values
aws["collect_dates_check"] = fi["dates_check"].values
new_year = aws["collect_start"].dt.year.astype("float")
# no date at all: keep the year parsed from the product name, as before
aws["collection_year"] = new_year.where(new_year.notna(), aws["collection_year_prev"])

changed = aws[aws["collection_year"] != aws["collection_year_prev"]]
changed[["name", "collection_year_prev", "collection_year", "collect_start", "collect_end", "collect_dates_check"]].to_csv(
    D / "collection_year_changes.csv", index=False
)
print(aws["collect_dates_check"].value_counts())
print(f"collection_year changed for {len(changed)} of {len(aws)} products")
print((aws["collection_year"] - aws["collection_year_prev"]).value_counts().sort_index())
aws.to_file(AWS, driver="GeoJSON")
