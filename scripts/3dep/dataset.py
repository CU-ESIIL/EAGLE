import torch
from torch.utils.data import Dataset, DataLoader
from numpy.lib.recfunctions import structured_to_unstructured
import numpy as np
import json
import pdal
from pyproj import CRS, Transformer
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info
from shapely.geometry import Point
import geopandas as gpd

EPT_STORAGE_CRS = "EPSG:3857"  # USGS public EPT tiles are indexed/stored in Web Mercator

ALS_TILES = gpd.read_file('../../datasets/USGS_3dep/usgs_3dep_resources_AWS.geojson').set_index('name')

def identity(x):
    return x

def get_utm_epsg(lon, lat):
    utm_crs_list = query_utm_crs_info(
        datum_name="WGS 84",
        area_of_interest=AreaOfInterest(
            west_lon_degree=lon, south_lat_degree=lat,
            east_lon_degree=lon, north_lat_degree=lat,
        ),
    )
    return int(utm_crs_list[0].code)


def load_als_cookie(
    als_tile_name,
    lat,
    lon,
    point_crs,
    radius=100,       # meters
    out_crs=None,     # default: local UTM
    n_segments=64,    # circle smoothness for the coarse EPT prefilter
):
    """Stream a cylindrical crop (radius around lat/lon) from an EPT tile, reprojected to UTM."""
    try:
        als_tile = ALS_TILES.loc[als_tile_name]
    except KeyError:
        raise ValueError(f"ALS tile '{als_tile_name}' not found in lookup.")

    if out_crs is None:
        out_crs = f"EPSG:{get_utm_epsg(lon, lat)}"

    # check that the point is within the tile's bounding box
    if not als_tile.geometry.contains(Point(lon, lat)):
        raise ValueError("Query point is outside the tile's bounding box.")

    # EPT bounds/polygon filtering happens in the tile's *storage* SRS (EPT_STORAGE_CRS),
    # not in `horiz_crs` (that field just describes the source point cloud's native CRS).
    # Use a padded circle here as a coarse prefilter -- distances in 3857 are distorted,
    # so this is not the exact cylinder, just a cheap way to avoid loading the whole tile.
    to_storage_crs = Transformer.from_crs(point_crs, EPT_STORAGE_CRS, always_xy=True)
    sx, sy = to_storage_crs.transform(lon, lat)
    prefilter_polygon = Point(sx, sy).buffer(radius * 1.5, quad_segs=n_segments).wkt

    # exact circle radius is applied after reprojecting to a true distance-preserving CRS (UTM)
    to_out_crs = Transformer.from_crs(point_crs, out_crs, always_xy=True)
    ux, uy = to_out_crs.transform(lon, lat)

    pipeline_json = {
        "pipeline": [
            {
                "type": "readers.ept",
                "filename": als_tile.url,
                "polygon": prefilter_polygon,   # coarse prefilter, in EPT storage CRS
                # "threads": 2,
            },
            {
                "type": "filters.range",
                "limits": "Classification![7:7]",  # drop noise
            },
            {
                "type": "filters.reprojection",
                "in_srs": EPT_STORAGE_CRS,
                "out_srs": out_crs,
            },
            {
                "type": "filters.crop",
                "point": f"POINT({ux} {uy})",
                "distance": radius,   # exact cylindrical trim, true meters in UTM
                # square geometry
                # "polygon": f"POLYGON(({ux-radius} {uy-radius}, {ux+radius} {uy-radius}, {ux+radius} {uy+radius}, {ux-radius} {uy+radius}, {ux-radius} {uy-radius}))",
            },
        ]
    }

    pipeline = pdal.Pipeline(json.dumps(pipeline_json))
    pipeline.execute(allowed_dims=["X", "Y", "Z", "Intensity", "ReturnNumber", "NumberOfReturns", "PointSourceId", "Classification"])
    return pipeline  # pipeline.arrays[0] -> structured numpy array in UTM meters


class ALSCookieDataset(Dataset):
    def __init__(self, site_table, radius=100):
        self.radius = radius
        self.site_table = site_table

    def __len__(self):
        return self.site_table.shape[0]

    def __getitem__(self, idx):
        # Randomly select a sample from the butterflies dataset
        sample = self.site_table.iloc[idx]
        pipeline = load_als_cookie(
            als_tile_name=sample['product_name_AWS'],
            lat=sample['Latitude'],
            lon=sample['Longitude'],
            point_crs='EPSG:4326',
            radius=self.radius,
            out_crs=None,
        )
        points = pipeline.arrays[0]
        # relative coordinates
        points['X'] = points['X'] - points['X'].min()
        points['Y'] = points['Y'] - points['Y'].min()

        # 1. Convert structured array to a flat unstructured float64 array
        #[['X', 'Y', 'Z', 'Intensity', 'ReturnNumber', 'NumberOfReturns', 'PointSourceId', 'Classification']]
        unstructured_arr = structured_to_unstructured(points, dtype=np.float64)

        return torch.from_numpy(unstructured_arr).float()