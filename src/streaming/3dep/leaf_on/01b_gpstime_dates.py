"""Step 1b: measure acquisition dates from the point clouds themselves (GpsTime of the EPT root node).

WESM dates are reported per workunit and reach us through a fragile name match; the GpsTime stored in
every point is the ground truth. The EPT root node (ept-data/0-0-0-0.laz, tens of KB) holds a thinned
sample of points spread over the whole project, so its GpsTime spread approximates the acquisition span
(extremes can be slightly narrower than the true span).

GpsTime conventions handled: Adjusted Standard GPS (value + 1e9 seconds since 1980-01-06, the norm),
Standard GPS (value >= 1e9) and GPS week seconds (< 604800, unrecoverable -> flagged, no date).
Output: datasets/USGS_3dep/leaf_on/gpstime_dates.csv  (name, gps_start, gps_p01, gps_median, gps_p99,
gps_end, gps_n_points, gps_status). Resumable.
"""

import concurrent.futures as cf
import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pdal
import requests

REPO = Path(__file__).resolve().parents[4]
AWS = REPO / "datasets/USGS_3dep/usgs_3dep_resources_AWS.geojson"
OUT = REPO / "datasets/USGS_3dep/leaf_on/gpstime_dates.csv"
GPS_EPOCH = datetime(1980, 1, 6)


def to_dates(g):
    g = g[np.isfinite(g)]
    if len(g) == 0 or g.max() < 604800 * 1.5:
        return None
    sec = np.where(g >= 1e9, g, g + 1e9)
    return [GPS_EPOCH + timedelta(seconds=float(s)) for s in np.percentile(sec, [0, 1, 50, 99, 100])]


def scan(args):
    name, url = args
    row = dict(name=name, gps_status="")
    base = url.rsplit("/", 1)[0]
    try:
        r = requests.get(f"{base}/ept-data/0-0-0-0.laz", timeout=60)
        if r.status_code != 200:
            row["gps_status"] = f"http_{r.status_code}"
            return row
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "root.laz"
            f.write_bytes(r.content)
            p = pdal.Pipeline(json.dumps([{"type": "readers.las", "filename": str(f)}]))
            p.execute()
            a = p.arrays[0]
        if "GpsTime" not in a.dtype.names:
            row["gps_status"] = "no_gpstime"
            return row
        d = to_dates(a["GpsTime"].astype(float))
        if d is None:
            row["gps_status"] = "gps_week_seconds_or_empty"
            return row
        row.update(zip(["gps_start", "gps_p01", "gps_median", "gps_p99", "gps_end"], [x.strftime("%Y-%m-%d") for x in d]))
        row["gps_n_points"] = len(a)
        row["gps_status"] = "ok" if 1999 <= d[2].year <= datetime.now().year else "implausible_year"
    except Exception as e:  # keep going; failures are recorded and retried on rerun
        row["gps_status"] = f"error_{type(e).__name__}"
    return row


def main():
    aws = gpd.read_file(AWS)[["name", "url"]]
    done = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame(columns=["name", "gps_status"])
    keep = done[~done["gps_status"].fillna("").str.startswith(("error", "http_5"))]
    todo = [(n, u) for n, u in zip(aws["name"], aws["url"]) if n not in set(keep["name"])]
    print(f"{len(todo)} products to scan")
    rows = []
    with cf.ThreadPoolExecutor(12) as ex:
        for i, row in enumerate(ex.map(scan, todo), 1):
            rows.append(row)
            if i % 200 == 0 or i == len(todo):
                pd.concat([keep, pd.DataFrame(rows)]).to_csv(OUT, index=False)
                print(f"{i}/{len(todo)}", flush=True)
    print(pd.read_csv(OUT)["gps_status"].value_counts())


if __name__ == "__main__":
    main()
