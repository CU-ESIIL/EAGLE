"""Optional: check that ALS can be streamed for rows of the NLCD eval table.

Streams a small crop per point from the 3DEP EPT and records the point count. Nothing is
cached or saved except the status table. Re-runnable: points already in the status file
are skipped, so it can be stopped at any time (`--max-seconds`); the full availability
mask is meant to be generated on cluster compute.

    python scripts/nlcd_eval/02_check_streaming.py --max-seconds 600 --workers 8
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pdal
from pyproj import Transformer
from shapely.geometry import Point

ROOT = Path(__file__).resolve().parents[2]
TABLE = ROOT / "datasets/NLCD_eval/nlcd_lidar_eval.parquet"
STATUS = ROOT / "datasets/NLCD_eval/nlcd_lidar_eval_als_status.csv"
EPT_CRS = "EPSG:3857"
RADIUS = 50  # m; covers the 30 m pixel plus context


def count_points(url, lat, lon):
    sx, sy = Transformer.from_crs(4326, EPT_CRS, always_xy=True).transform(lon, lat)
    poly = Point(sx, sy).buffer(RADIUS * 1.5).wkt  # coarse prefilter in web mercator
    pipe = pdal.Pipeline(json.dumps({"pipeline": [
        {"type": "readers.ept", "filename": url, "polygon": poly},
        {"type": "filters.range", "limits": "Classification![7:7]"},
    ]}))
    pipe.execute(allowed_dims=["X", "Y", "Z", "Classification"])
    return sum(len(a) for a in pipe.arrays)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-seconds", type=float, default=600)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--all", action="store_true", help="check all rows, not just balanced_subset")
    a = ap.parse_args()

    t = pd.read_parquet(TABLE)
    if not a.all:
        t = t[t.balanced_subset]
    done = pd.read_csv(STATUS) if STATUS.exists() else pd.DataFrame(columns=["als_site_id"])
    t = t[~t.als_site_id.isin(done.als_site_id)]
    urls = gpd.read_file(ROOT / "datasets/USGS_3dep/usgs_3dep_resources_AWS.geojson").set_index("name").url
    # shuffle so that a partial run is an unbiased sample
    t = t.sample(frac=1, random_state=0)

    start, out = time.time(), []

    def work(r):
        try:
            n = count_points(urls[r.product_name_AWS], r.lat, r.lon)
            return dict(als_site_id=r.als_site_id, product_name_AWS=r.product_name_AWS, lat=r.lat, lon=r.lon,
                        status="ok" if n > 0 else "empty", n_points=n, error="")
        except Exception as e:  # noqa: BLE001
            return dict(als_site_id=r.als_site_id, product_name_AWS=r.product_name_AWS, lat=r.lat, lon=r.lon,
                        status="error", n_points=0, error=str(e)[:200])

    # work in small batches and append after each, so a crash (PDAL can abort the whole
    # process on a transient S3 error) loses at most one batch; just re-run to resume
    rows = list(t.itertuples())
    n_done = 0
    with ThreadPoolExecutor(a.workers) as ex:
        for i in range(0, len(rows), 4 * a.workers):
            if time.time() - start > a.max_seconds:
                break
            batch = pd.DataFrame(list(ex.map(work, rows[i : i + 4 * a.workers])))
            batch.to_csv(STATUS, mode="a", header=not STATUS.exists(), index=False)
            out.append(batch)
            n_done += len(batch)
    new = pd.concat(out) if out else pd.DataFrame(columns=["status"])
    print(f"checked {n_done} in {time.time()-start:.0f}s;", new.status.value_counts().to_dict())


if __name__ == "__main__":
    main()
