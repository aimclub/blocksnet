import shapely
import geopandas as gpd
from loguru import logger
from blocksnet.config import log_config
from .schemas import BlocksSchema

OUTER_RADIUS_COLUMN = "outer_radius"
INNER_RADIUS_COLUMN = "inner_radius"


def _calculate_outer_radius(polygon: shapely.Polygon) -> float:
    return shapely.minimum_bounding_radius(polygon)


def _calculate_inner_radius(polygon: shapely.Polygon) -> float:
    circle_radius = shapely.maximum_inscribed_circle(polygon)
    return circle_radius.length


def calculate_radiuses(blocks_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Calculate inner and outer radiuses for block polygons.

    Parameters
    ----------
    blocks_gdf : geopandas.GeoDataFrame
        GeoDataFrame validated by :class:`BlocksSchema` that contains polygon
        geometries.

    Returns
    -------
    geopandas.GeoDataFrame
        Copy of the validated dataframe with ``outer_radius`` and
        ``inner_radius`` columns describing circumradius and inradius
        approximations.

    Raises
    ------
    ValueError
        If ``blocks_gdf`` fails schema validation.
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
