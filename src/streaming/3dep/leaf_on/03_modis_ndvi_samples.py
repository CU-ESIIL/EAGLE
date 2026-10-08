"""Step 3: sample MODIS 16-day NDVI (MOD13Q1, 250 m) at points inside each product for the collection year(s).

External phenology source: Terra MOD13Q1 v061 COGs on Microsoft Planetary Computer (anonymous, signed
STAC hrefs). For every product with collection dates we draw N_POINTS random points inside its
footprint (deterministic per product) and read the NDVI of every 16-day composite in each calendar year
the collection touches. Each COG is opened once and all points that fall inside its tile are sampled.
(MCD12Q2 land surface phenology would be the natural product but it is not on Planetary Computer and
needs an Earthdata login.)

Output: datasets/USGS_3dep/leaf_on/modis_ndvi_samples.parquet  (name, pt, lon, lat, year, doy, ndvi)
Resumable per year (partial years are re-run). Usage: python -I 03_modis_ndvi_samples.py [--limit N] [--years 2019,2020]
"""

import argparse
import time
import concurrent.futures as cf
import zlib
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import planetary_computer as pc
import pystac_client
import rasterio
from rasterio.warp import transform
from shapely.geometry import MultiPoint, Point, shape

REPO = Path(__file__).resolve().parents[4]
AWS = REPO / "datasets/USGS_3dep/usgs_3dep_resources_AWS.geojson"
DATES = REPO / "datasets/USGS_3dep/leaf_on/collect_dates_final.csv"
OUT = REPO / "datasets/USGS_3dep/leaf_on/modis_ndvi_samples.parquet"
N_POINTS = 8
WORKERS = 16

catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")


def sample_points(name, geom, n=N_POINTS):
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    pts = [geom.representative_point()]
    minx, miny, maxx, maxy = geom.bounds
    tries = 0
    while len(pts) < n and tries < 2000:
        tries += 1
        p = Point(rng.uniform(minx, maxx), rng.uniform(miny, maxy))
        if geom.contains(p):
            pts.append(p)
    return [(p.x, p.y) for p in pts]


def read_item(item, pts, retries=3):
    """Transient COG read errors are retried; an item that keeps failing is skipped (logged)."""
    for attempt in range(retries):
        try:
            return _read_item(item, pts)
        except Exception as e:
            if attempt == retries - 1:
                print(f"skip {item.id}: {type(e).__name__}", flush=True)
                return None
            time.sleep(2**attempt)


def _read_item(item, pts):
    """pts: DataFrame[name, pt, lon, lat]; returns rows for points inside this tile."""
    href = pc.sign(item.assets["250m_16_days_NDVI"].href)
    with rasterio.open(href) as ds:
        x, y = transform("EPSG:4326", ds.crs, pts["lon"].tolist(), pts["lat"].tolist())
        b = ds.bounds
        inside = [(b.left <= xi < b.right) and (b.bottom < yi <= b.top) for xi, yi in zip(x, y)]
        sel = pts[inside]
        if sel.empty:
            return None
        xy = [(xi, yi) for xi, yi, k in zip(x, y, inside) if k]
        vals = np.array([v[0] for v in ds.sample(xy)], dtype=float)
        nodata, scale = ds.nodata, ds.scales[0]
    vals[vals == nodata] = np.nan
    vals = vals / scale
    date = pd.Timestamp(item.properties["start_datetime"][:10])
    out = sel.copy()
    out["year"], out["doy"], out["ndvi"] = date.year, date.dayofyear, vals
    return out


def run_year(year, pts):
    geom = MultiPoint(list(zip(pts["lon"], pts["lat"])))
    search = catalog.search(
        collections=["modis-13Q1-061"],
        intersects=geom.__geo_interface__,
        datetime=f"{year}-01-01/{year}-12-31",
        query={"platform": {"eq": "terra"}},
    )
    items = list(search.items())
    tile_geoms = {i.id: shape(i.geometry) for i in items}
    jobs = []
    for it in items:
        near = pts[[tile_geoms[it.id].buffer(0.2).contains(Point(lo, la)) for lo, la in zip(pts["lon"], pts["lat"])]]
        if len(near):
            jobs.append((it, near))
    print(f"{year}: {len(pts)} points, {len(items)} items, {len(jobs)} reads", flush=True)
    rows = []
    with cf.ThreadPoolExecutor(WORKERS) as ex:
        for res in ex.map(lambda j: read_item(*j), jobs):
            if res is not None:
                rows.append(res)
    return pd.concat(rows) if rows else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--years")
    args = ap.parse_args()

    aws = gpd.read_file(AWS).set_index("name")
    dates = pd.read_csv(DATES, parse_dates=["collect_start", "collect_end"]).dropna(subset=["collect_start"])
    if args.limit:
        dates = dates.sample(args.limit, random_state=0)
    # one row per (product, year touched by the collection window)
    need = []
    for r in dates.itertuples():
        for y in range(r.collect_start.year, r.collect_end.year + 1):
            need.append((r.name, y))
    need = pd.DataFrame(need, columns=["name", "year"])
    if args.years:
        need = need[need.year.isin([int(y) for y in args.years.split(",")])]

    done = pd.read_parquet(OUT) if OUT.exists() else pd.DataFrame(columns=["name", "year"])
    done_keys = set(zip(done["name"], done["year"]))
    need = need[[k not in done_keys for k in zip(need["name"], need["year"])]]
    print(f"{len(need)} product-years to sample")

    chunks = [done]
    for year, grp in need.groupby("year"):
        recs = []
        for name in grp["name"]:
            for k, (lo, la) in enumerate(sample_points(name, aws.loc[name, "geometry"])):
                recs.append((name, k, lo, la))
        pts = pd.DataFrame(recs, columns=["name", "pt", "lon", "lat"])
        res = run_year(year, pts)
        # mark product-years with no valid reads too, so they are not retried forever
        if len(res):
            chunks.append(res)
        pd.concat(chunks).to_parquet(OUT)
        print(f"{year}: saved, total rows {sum(len(c) for c in chunks)}", flush=True)


if __name__ == "__main__":
    main()
