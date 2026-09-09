import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path
# metadata table of available USGS 3DEP 

resources_aws = gpd.read_file(Path(__file__).parent / "usgs_3dep_resources_AWS.geojson")
resources_usgs = gpd.read_file(Path(__file__).parent / "usgs_3dep_resources_ALL.geojson")

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
