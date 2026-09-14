import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path
# metadata table of available USGS 3DEP 

data_root = Path(__file__).parent.parent.parent.parent / "datasets/USGS_3dep"
resources_aws = gpd.read_file(data_root / "usgs_3dep_resources_AWS.geojson")
resources_usgs = gpd.read_file(data_root / "usgs_3dep_resources_ALL.geojson")

def lookup_3dep_by_latlon(lat, lon, earliest_year=None, latest_year=None, source="AWS"):
    """
    Lookup USGS 3DEP resources by latitude and longitude.
    
    Parameters:
    lat (float): Latitude of the point (CRS: EPSG:4326).
    lon (float): Longitude of the point (CRS: EPSG:4326).
    earliest_year (int, optional): Earliest collection year to filter resources. Defaults to None.
    latest_year (int, optional): Latest collection year to filter resources. Defaults to None.

    Returns:
    pd.DataFrame: A DataFrame containing the matching 3DEP LAS data products, which can be downloaded via PDAL
    """
    pt = Point(lon, lat)
    if source == "AWS":
        resources = resources_aws
    elif source == "USGS":
        resources = resources_usgs
    else:
        raise ValueError("Invalid source. Must be 'AWS' or 'USGS'.")
    matches = resources[resources.geometry.contains(pt)]
    if earliest_year is not None:
        matches = matches[matches['collection_year'] >= earliest_year]
    if latest_year is not None:
        matches = matches[matches['collection_year'] <= latest_year]
    return matches

import functools

@functools.lru_cache()
def get_nearest_year_product(lat, lon, year, source="AWS"):
    """
    Get the nearest 3DEP product for a given latitude, longitude, and year.
    
    Parameters:
    lat (float): Latitude of the point (CRS: EPSG:4326).
    lon (float): Longitude of the point (CRS: EPSG:4326).
    year (int): Year to find the nearest product.

    Returns:
    pd.Series: A Series containing the nearest 3DEP LAS data product information,
    or None if no products are found.
    """
    matches = lookup_3dep_by_latlon(lat, lon, source=source)
    if matches.empty:
        return None
    matches['year_diff'] = matches['collection_year'].apply(lambda x: abs(x - year))
    nearest = matches.loc[matches['year_diff'].idxmin()]

    return {
        f"product_name_{source}": nearest['name'],
        f"collection_year_{source}": int(nearest['collection_year']),
        f"year_diff_{source}": int(nearest['year_diff']),
    }