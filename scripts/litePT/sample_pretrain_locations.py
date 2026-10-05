"""Sample random locations inside USGS 3DEP (AWS EPT) tile footprints for self-supervised pre-training.

Tiles are drawn with probability proportional to footprint_area ** --area-power (0.5 by default),
so large statewide collections don't dominate while small project tiles are still represented.
Each location is assigned to the tile it was drawn from (column product_name_AWS).

    source scripts/litePT/env.sh
    python scripts/litePT/sample_pretrain_locations.py --n 60000 --out $EAGLE_SCRATCH/pretrain/locations_v1.parquet

Optionally exclude locations near evaluation sites (e.g. vegbank plots) with --exclude table.parquet
(--exclude-lat-col/--exclude-lon-col) and --exclude-buffer meters.
"""

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from eagle_als.fetch import TILE_INDEX
from eagle_als.sites import add_site_ids, read_table


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=60000)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--area-power", type=float, default=0.5)
    p.add_argument("--min-year", type=int, default=None, help="drop tiles collected before this year")
    p.add_argument("--exclude", default=None)
    p.add_argument("--exclude-lat-col", default="lat")
    p.add_argument("--exclude-lon-col", default="lon")
    p.add_argument("--exclude-buffer", type=float, default=500.0)
    a = p.parse_args()

    rng = np.random.default_rng(a.seed)
    tiles = gpd.read_file(TILE_INDEX)
    tiles = tiles[tiles["url"].notna() & tiles.geometry.notna() & ~tiles.geometry.is_empty]
    if a.min_year is not None:
        tiles = tiles[tiles.collection_year >= a.min_year]
    area_km2 = tiles.to_crs("EPSG:6933").area / 1e6  # equal-area projection
    w = area_km2.to_numpy() ** a.area_power
    w = w / w.sum()
    counts = rng.multinomial(a.n, w)
    print(f"{len(tiles)} tiles, {(counts > 0).sum()} receive >=1 location")

    rows = []
    for (_, tile), k in zip(tiles.iterrows(), counts):
        if k == 0:
            continue
        pts = gpd.GeoSeries([tile.geometry], crs=tiles.crs).sample_points(int(k), rng=rng).explode()
        for pt in pts:
            rows.append(dict(lat=pt.y, lon=pt.x, product_name_AWS=tile["name"],
                             collection_year_AWS=tile.collection_year))
    df = pd.DataFrame(rows)

    if a.exclude:
        ex = read_table(a.exclude)
        ex = gpd.GeoDataFrame(ex, geometry=gpd.points_from_xy(ex[a.exclude_lon_col], ex[a.exclude_lat_col]), crs=4326)
        pts = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs=4326)
        near = gpd.sjoin_nearest(pts.to_crs(5070), ex.to_crs(5070)[["geometry"]], max_distance=a.exclude_buffer, how="inner")
        drop = near.index.unique()
        print(f"excluding {len(drop)} locations within {a.exclude_buffer} m of {a.exclude}")
        df = df.drop(index=drop)

    df = add_site_ids(df, "lat", "lon").sample(frac=1.0, random_state=a.seed).reset_index(drop=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(a.out, index=False)
    print(f"wrote {len(df)} locations -> {a.out}")


if __name__ == "__main__":
    main()
