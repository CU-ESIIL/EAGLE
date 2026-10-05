"""Helpers for site tables (labeled datasets and pre-training locations)."""

import re
from pathlib import Path

import pandas as pd


def read_table(path):
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix in (".gpkg", ".geojson", ".shp"):
        import geopandas as gpd

        return gpd.read_file(path)
    return pd.read_csv(path, low_memory=False)


def site_id(tile, lat, lon):
    """Stable id for a (3DEP tile, location) pair; ~1 m precision."""
    tile = re.sub(r"[^A-Za-z0-9_-]", "_", str(tile))
    return f"{tile}__{lat:.5f}_{lon:.5f}"


def add_site_ids(df, lat_col, lon_col, tile_col="product_name_AWS"):
    df = df.copy()
    ok = df[tile_col].notna() & df[lat_col].notna() & df[lon_col].notna()
    df["als_site_id"] = None
    df.loc[ok, "als_site_id"] = [
        site_id(t, la, lo) for t, la, lo in zip(df.loc[ok, tile_col], df.loc[ok, lat_col], df.loc[ok, lon_col])
    ]
    return df
