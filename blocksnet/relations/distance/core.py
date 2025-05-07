import geopandas as gpd
import numpy as np
from scipy.spatial import distance_matrix
from .schemas import BlocksSchema


def calculate_distance_matrix(blocks_gdf: gpd.GeoDataFrame, dtype="int32") -> np.ndarray:
    blocks_gdf = BlocksSchema(blocks_gdf)
    xs = blocks_gdf.geometry.x
    ys = blocks_gdf.geometry.y
    coords = np.array(list(zip(xs, ys)))
    matrix = distance_matrix(coords, coords)
    return gpd.GeoDataFrame(np.round(matrix), index=blocks_gdf.index, columns=blocks_gdf.index, dtype=dtype)
