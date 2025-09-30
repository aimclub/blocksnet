import geopandas as gpd
from .schemas import BlocksSchema

SITE_AREA_COLUMN = "site_area"
SITE_LENGTH_COLUMN = "site_length"


def calculate_area_length(blocks_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Calculate area length.

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
    blocks_gdf[SITE_AREA_COLUMN] = blocks_gdf.area
    blocks_gdf[SITE_LENGTH_COLUMN] = blocks_gdf.length
    return blocks_gdf
