import geopandas as gpd
from shapely import Polygon
from .schemas import BlocksSchema
from blocksnet.config import log_config

CENTERLINE_LENGTH_COLUMN = "centerline_length"


def calculate_centerlines(blocks_gdf: gpd.GeoDataFrame):
    """Estimate centerline length for each block polygon.

    Parameters
    ----------
    blocks_gdf : geopandas.GeoDataFrame
        GeoDataFrame satisfying :class:`BlocksSchema` that provides polygon
        geometries for each block.

    Returns
    -------
    geopandas.GeoDataFrame
        Copy of the validated dataframe with a ``centerline_length`` column
        describing the medial axis length for every polygon.

    Raises
    ------
    ImportError
        If :mod:`pygeoops` is not installed.
    ValueError
        If schema validation fails for ``blocks_gdf``.
    """
    try:
        import pygeoops
    except ImportError:
        raise ImportError("PyGeoOps package is required but not installed")

    blocks_gdf = BlocksSchema(blocks_gdf)

    def _calculate_centerline(polygon: Polygon) -> float:
        try:
            return pygeoops.centerline(polygon).length
        except Exception as e:
            return 0.0

    if log_config.disable_tqdm:
        centerline = blocks_gdf.geometry.apply(_calculate_centerline)
    else:
        centerline = blocks_gdf.geometry.progress_apply(_calculate_centerline)

    blocks_gdf[CENTERLINE_LENGTH_COLUMN] = centerline

    return blocks_gdf
