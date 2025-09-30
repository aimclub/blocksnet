import geopandas as gpd
import numpy as np
from scipy.spatial import distance_matrix
from .schemas import BlocksSchema


def calculate_distance_matrix(blocks_gdf: gpd.GeoDataFrame, dtype="int32") -> np.ndarray:
    """Compute pairwise Euclidean distances between block centroids.

    Parameters
    ----------
    blocks_gdf : geopandas.GeoDataFrame
        Blocks with polygon geometries to measure distances for. Input is
        validated with :class:`BlocksSchema`.
    dtype : str or numpy.dtype, default="int32"
        Data type for the resulting distance matrix.

    Returns
    -------
    geopandas.GeoDataFrame
        Symmetric matrix of rounded centroid-to-centroid distances where rows
        and columns align with ``blocks_gdf.index``.
    """

    blocks_gdf = BlocksSchema(blocks_gdf)
    xs = blocks_gdf.geometry.x
    ys = blocks_gdf.geometry.y
    coords = np.array(list(zip(xs, ys)))
    matrix = distance_matrix(coords, coords)
    return gpd.GeoDataFrame(np.round(matrix), index=blocks_gdf.index, columns=blocks_gdf.index, dtype=dtype)
