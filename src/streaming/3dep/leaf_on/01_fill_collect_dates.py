"""Step 1: recover collection dates for AWS EPT products that failed the exact WESM name merge.

fetch_3dep_metadata.py merges WESM on workunit == EPT folder name, which only works for 1,033 of the
2,278 products. The rest are mostly bucket names like 'USGS_LPC_SD_NRCS_DAS_2017_LAS_2019' (the
trailing year is the publication year, not the collection year) or differ from WESM in case,
punctuation or an 'ARRA-' prefix. Here we match on a normalized name against WESM workunit and project
and take the earliest start / latest end over all matching rows.

Output: datasets/USGS_3dep/leaf_on/collect_dates.csv  (one row per product name; also carries the
matched WESM metadata_link(s), which step 2 scrapes for provider leaf-on/off statements)
"""

import re
from pathlib import Path

import geopandas as gpd
import pandas as pd

REPO = Path(__file__).resolve().parents[4]
AWS = REPO / "datasets/USGS_3dep/usgs_3dep_resources_AWS.geojson"
WESM = REPO / "scripts/3dep/WESM.csv"
OUT = REPO / "datasets/USGS_3dep/leaf_on/collect_dates.csv"


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def strip_bucket_decoration(name):
    name = re.sub(r"^USGS_(LPC|Lidar_Point_Cloud)_+", "", name)
    return re.sub(r"_+LAS_\d{4}$", "", name)


aws = gpd.read_file(AWS)
wesm = pd.read_csv(WESM, parse_dates=["collect_start", "collect_end"])

# normalized key -> row indices in WESM, separately for workunit and project names
index = {"workunit": {}, "project": {}}
for col in index:
    for i, v in wesm[col].items():
        if isinstance(v, str):
            for k in {norm(v), norm(re.sub(r"_Legacy_Data$", "", v))}:
                index[col].setdefault(k, []).append(i)

rows = []


def links(sub):
    return "|".join(sub["metadata_link"].dropna().unique()[:5])


for name, start, end in zip(aws["name"], aws["collect_start"], aws["collect_end"]):
    if pd.notna(start):
        rows.append((name, start, end, "wesm_exact_workunit", links(wesm[wesm.workunit == name])))
        continue
    base = strip_bucket_decoration(name)
    keys = [norm(base), norm(re.sub(r"^ARRA-?", "", base))]
    hit, source = None, None
    for col in ("workunit", "project"):
        for k in keys:
            if k in index[col]:
                hit, source = index[col][k], f"wesm_normalized_{col}"
                break
        if hit:
            break
    if hit is None:
        rows.append((name, pd.NaT, pd.NaT, None, ""))
    else:
        sub = wesm.loc[hit]
        rows.append((name, sub.collect_start.min(), sub.collect_end.max(), source, links(sub)))

out = pd.DataFrame(rows, columns=["name", "collect_start", "collect_end", "dates_source", "metadata_links"])
out["collect_start"] = pd.to_datetime(out["collect_start"]).dt.strftime("%Y-%m-%d")
out["collect_end"] = pd.to_datetime(out["collect_end"]).dt.strftime("%Y-%m-%d")
out.to_csv(OUT, index=False)

print(out["dates_source"].value_counts(dropna=False))
yr = pd.to_datetime(out["collect_start"]).dt.year
old = aws.set_index("name")["collection_year"].reindex(out["name"]).values
diff = (yr.values != old) & yr.notna().values & pd.notna(old)
print(f"collection_year in registry disagrees with recovered start year for {diff.sum()} products")
