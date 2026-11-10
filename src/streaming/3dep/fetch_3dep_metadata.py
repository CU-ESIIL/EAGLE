import time

import geopandas as gpd
import pandas as pd
import requests

# hobu's resources.geojson is a live mirror of the s3://usgs-lidar-public EPT bucket:
# 'name' is the bucket folder basename and 'url' is built directly from it
# (base_url + name + '/ept.json'), so url is reliable by construction and the file
# tracks new EPT builds (confirmed includes 2026-built _D24 deliveries).
# However, EPT currently lags USGS 3DEP by nearly 2 years: no 2025 tiles, few 2024 tiles in Sept 2026
resources = gpd.read_file(
    "https://raw.githubusercontent.com/hobu/usgs-lidar/master/boundaries/resources.geojson"
)

# WESM.csv has richer metadata (collect dates, QL, spec) than hobu provides.
# The official WESM.gpkg download hangs (as of Sept 2026); the csv is the reliable source.
wesm_metadata = pd.read_csv(
    "https://rockyweb.usgs.gov/vdelivery/Datasets/Staged/Elevation/metadata/WESM.csv",
    parse_dates=["collect_start", "collect_end"],
)
resources = resources.merge(wesm_metadata, how="left", left_on="name", right_on="workunit")


def get_year(row):
    if row["collect_start"] is not pd.NaT:
        return row["collect_start"].year
    elif row["collect_end"] is not pd.NaT:
        return row["collect_end"].year
    else:
        try:
            return int(row["name"][-4:])
        except ValueError:
            return None


# all files have year in the name, some don't have the date
resources["collection_year"] = resources.apply(get_year, axis=1)

resources.to_file("usgs_3dep_resources_AWS.geojson", driver="GeoJSON")

# Version that contains all workunits published by USGS and their direct staged-LAZ
# download links (lpc_link), rather than only the subset available via AWS EPT above.
# Queries the National Map ArcGIS REST index directly (same WESM data as the .gpkg/.csv,
# but paginated so it doesn't hang the way the WESM.gpkg download does).
WESM_LIDAR_LAYER_URL = (
    "https://index.nationalmap.gov/arcgis/rest/services/3DEPElevationIndex/MapServer/8/query"
)


def fetch_wesm_lidar_boundaries(page_size=500, timeout=60, max_retries=5):
    """Fetch all lidar workunit boundaries + metadata from the National Map WESM index layer."""
    count_resp = requests.get(
        WESM_LIDAR_LAYER_URL,
        params={"where": "1=1", "returnCountOnly": "true", "f": "json"},
        timeout=timeout,
    )
    count_resp.raise_for_status()
    total = count_resp.json()["count"]

    frames = []
    for offset in range(0, total, page_size):
        params = {
            "where": "1=1",
            "outFields": "*",
            "f": "geojson",
            "resultOffset": offset,
            "resultRecordCount": page_size,
        }
        # the endpoint intermittently 500s regardless of page size; retry with backoff
        for attempt in range(max_retries):
            resp = requests.get(WESM_LIDAR_LAYER_URL, params=params, timeout=timeout)
            if resp.ok:
                break
            if attempt == max_retries - 1:
                resp.raise_for_status()
            time.sleep(2**attempt)
        frames.append(gpd.GeoDataFrame.from_features(resp.json()["features"], crs="EPSG:4326"))
        print(f"fetched {min(offset + page_size, total)}/{total}")

    return pd.concat(frames, ignore_index=True)


usgs_resources = fetch_wesm_lidar_boundaries()
usgs_resources = usgs_resources.rename(columns={"workunit": "name"})

for col in ("collect_start", "collect_end", "lpc_pub_date"):
    usgs_resources[col] = pd.to_datetime(usgs_resources[col], unit="ms", errors="coerce")

usgs_resources["collection_year"] = usgs_resources.apply(get_year, axis=1)

usgs_resources.to_file("usgs_3dep_resources_ALL.geojson", driver="GeoJSON")
