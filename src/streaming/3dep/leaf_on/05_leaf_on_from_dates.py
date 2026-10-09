"""Step 5: set leaf_on from the collection dates alone.

NDVI is a poor proxy for deciduous leaf state (evergreen conifers stay green in winter), and collections that span
months or fall in shoulder seasons cannot be given a binary label. CONUS only, so a simple calendar rule is used:

  leaf_on = True   every month touched by [collect_start, collect_end] is in June-September
  leaf_on = False  every month touched is in December-March
  leaf_on = null   anything else (spans other months, shoulder season, or no dates)

Sets leaf_on and leaf_on_source ('collection_season' or null) on datasets/USGS_3dep/usgs_3dep_resources_AWS.geojson.
Usage: python -I 05_leaf_on_from_dates.py
"""

from pathlib import Path

import geopandas as gpd
import pandas as pd

REPO = Path(__file__).resolve().parents[4]
AWS = REPO / "datasets/USGS_3dep/usgs_3dep_resources_AWS.geojson"
LEAF_ON_MONTHS = {6, 7, 8, 9}
LEAF_OFF_MONTHS = {12, 1, 2, 3}


def season_label(start, end):
    if pd.isna(start) or pd.isna(end):
        return None
    months = {p.month for p in pd.period_range(start, end, freq="M")}
    if months <= LEAF_ON_MONTHS:
        return True
    if months <= LEAF_OFF_MONTHS:
        return False
    return None


def main():
    aws = gpd.read_file(AWS)
    for c in ["collect_start", "collect_end"]:
        aws[c] = pd.to_datetime(aws[c])
    lab = [season_label(s, e) for s, e in zip(aws["collect_start"], aws["collect_end"])]
    aws["leaf_on"] = pd.array(lab, dtype="boolean")
    aws["leaf_on_source"] = pd.Series(["collection_season" if x is not None else None for x in lab], dtype=object)
    print(aws["leaf_on"].value_counts(dropna=False))
    aws.to_file(AWS, driver="GeoJSON")


if __name__ == "__main__":
    main()
