import geopandas as gpd
from .schemas import BlocksSchema

SITE_AREA_COLUMN = "site_area"
SITE_LENGTH_COLUMN = "site_length"


def calculate_area_length(blocks_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Calculate polygon areas and perimeter lengths for blocks.

    Parameters
    ----------
    blocks_gdf : geopandas.GeoDataFrame
        GeoDataFrame that satisfies :class:`BlocksSchema` and contains block
        geometries in projected coordinates.

    Returns
    -------
    geopandas.GeoDataFrame
        Copy of the validated dataframe with ``site_area`` and ``perimeter``
        columns computed from the geometries.

    Raises
    ------
    ValueError
        If ``blocks_gdf`` fails schema validation.
    """
    blocks_gdf = BlocksSchema(blocks_gdf)
    blocks_gdf[SITE_AREA_COLUMN] = blocks_gdf.area
    blocks_gdf[SITE_LENGTH_COLUMN] = blocks_gdf.length
    return blocks_gdf
