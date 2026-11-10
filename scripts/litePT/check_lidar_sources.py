"""Check that every site in a labeled dataset has usable 3DEP lidar, and write a filtered table.

For each unique (tile, lat, lon) the cookie is fetched with PDAL and cached locally
(this cache is re-used by training/evaluation). Sites whose fetch is empty, errors
(missing/unreadable EPT tile), has too few points, or fails validation are dropped.

    source scripts/litePT/env.sh
    python scripts/litePT/check_lidar_sources.py --dataset ofo
    python scripts/litePT/check_lidar_sources.py --dataset bbs --workers 24
    # any other table (e.g. vegbank, once ready):
    python scripts/litePT/check_lidar_sources.py --table path/to/vegbank.parquet \
        --lat-col lat --lon-col lon --tile-col product_name_AWS --name vegbank

Outputs:
    $EAGLE_SCRATCH/cache/sites/...            cached cookies + fetch manifest (shared by all datasets)
    <table dir>/<stem>_als_available.parquet  input rows with usable lidar (+ als_site_id, als_n_points)
    <table dir>/<stem>_als_status.csv         one row per unique site with fetch status / error
"""

import argparse
import os
from pathlib import Path

import pandas as pd

from eagle_als.cache import build_cache, read_manifest
from eagle_als.sites import add_site_ids, read_table

REPO = Path(__file__).resolve().parents[2]
PRESETS = {
    "bbs": dict(table=REPO / "datasets/BBS/BBS_train_test.parquet", lat_col="Latitude", lon_col="Longitude"),
    "butterflies": dict(table=REPO / "datasets/butterflies/species_observations.csv", lat_col="lat", lon_col="lon"),
    "ofo": dict(table=REPO / "datasets/OFO_trees/plots_w_als.gpkg", lat_col="plot_lat", lon_col="plot_lon"),
}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=sorted(PRESETS))
    p.add_argument("--table", help="table (csv/parquet/gpkg) with one row per labeled sample")
    p.add_argument("--name", help="dataset name for messages")
    p.add_argument("--lat-col", default="lat")
    p.add_argument("--lon-col", default="lon")
    p.add_argument("--tile-col", default="product_name_AWS")
    p.add_argument("--cache", default=None, help="default: $EAGLE_SCRATCH/cache/sites")
    p.add_argument("--radius", type=float, default=100.0)
    p.add_argument("--min-points", type=int, default=1000, help="sites with fewer points are dropped")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--retry-errors", action="store_true", help="re-attempt sites that previously errored")
    p.add_argument("--out-dir", default=None, help="default: next to the input table")
    a = p.parse_args()

    if a.dataset:
        cfg = PRESETS[a.dataset]
        table, lat_col, lon_col, name = Path(cfg["table"]), cfg["lat_col"], cfg["lon_col"], a.dataset
    else:
        table, lat_col, lon_col, name = Path(a.table), a.lat_col, a.lon_col, a.name or Path(a.table).stem
    cache = Path(a.cache or Path(os.environ["EAGLE_SCRATCH"]) / "cache/sites")

    df = read_table(table)
    df = add_site_ids(df, lat_col, lon_col, a.tile_col)
    sites = df.dropna(subset=["als_site_id"]).drop_duplicates("als_site_id")
    print(f"[{name}] {len(df)} rows, {len(sites)} unique sites, {df.als_site_id.isna().sum()} rows without a 3DEP tile")

    build_cache(
        sites, cache, id_col="als_site_id", lat_col=lat_col, lon_col=lon_col, tile_col=a.tile_col,
        radius=a.radius, min_points=1, workers=a.workers, retry_errors=a.retry_errors,
    )

    man = read_manifest(cache).rename(columns={"site_id": "als_site_id"})
    status = sites[["als_site_id", a.tile_col, lat_col, lon_col]].merge(
        man[["als_site_id", "status", "n_points", "error", "path"]], on="als_site_id", how="left"
    )
    status["status"] = status["status"].fillna("not_attempted")
    low = (status.status == "ok") & (status.n_points < a.min_points)
    status.loc[low, "status"] = "too_few_points"
    ok = status.loc[status.status == "ok", ["als_site_id", "n_points"]].rename(columns={"n_points": "als_n_points"})

    out_dir = Path(a.out_dir) if a.out_dir else table.parent
    stem = table.stem
    status.to_csv(out_dir / f"{stem}_als_status.csv", index=False)
    available = df.merge(ok, on="als_site_id", how="inner")
    if hasattr(available, "geometry") and "geometry" in available:
        available = pd.DataFrame(available.drop(columns="geometry"))
    available.to_parquet(out_dir / f"{stem}_als_available.parquet", index=False)

    print(f"\n[{name}] site status:\n{status.status.value_counts().to_string()}")
    print(f"[{name}] rows kept: {len(available)}/{len(df)} -> {out_dir / f'{stem}_als_available.parquet'}")
    errs = status.loc[status.status == "error", "error"].str.slice(0, 120).value_counts().head(10)
    if len(errs):
        print(f"[{name}] most common errors:\n{errs.to_string()}")


if __name__ == "__main__":
    main()
