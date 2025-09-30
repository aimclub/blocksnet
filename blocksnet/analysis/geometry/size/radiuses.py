import shapely
import geopandas as gpd
from loguru import logger
from blocksnet.config import log_config
from .schemas import BlocksSchema

OUTER_RADIUS_COLUMN = "outer_radius"
INNER_RADIUS_COLUMN = "inner_radius"


def _calculate_outer_radius(polygon: shapely.Polygon) -> float:
    """Calculate outer radius.

    Parameters
    ----------
    polygon : shapely.Polygon
        Description.

    Returns
    -------
    float
        Description.

    """
    return shapely.minimum_bounding_radius(polygon)


def _calculate_inner_radius(polygon: shapely.Polygon) -> float:
    """Calculate inner radius.

    Parameters
    ----------
    polygon : shapely.Polygon
        Description.

    Returns
    -------
    float
        Description.

    """
    circle_radius = shapely.maximum_inscribed_circle(polygon)
    return circle_radius.length


def calculate_radiuses(blocks_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Calculate radiuses.

    Parameters
    ----------
    blocks_gdf : gpd.GeoDataFrame
        Description.

    Returns
    -------
    gpd.GeoDataFrame
        Description.

    """
    blocks_gdf = BlocksSchema(blocks_gdf)

    if log_config.disable_tqdm:
        apply = blocks_gdf.geometry.apply
    else:
        apply = blocks_gdf.geometry.progress_apply

    outer_radius = apply(_calculate_outer_radius)
    inner_radius = apply(_calculate_inner_radius)

    blocks_gdf[OUTER_RADIUS_COLUMN] = outer_radius
    blocks_gdf[INNER_RADIUS_COLUMN] = inner_radius

    return blocks_gdf
