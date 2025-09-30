import geopandas as gpd
from .schemas import BlocksSchema

X_COLUMN = "x"
Y_COLUMN = "y"


def calculate_centroids(blocks_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Create centroid geometry for each block polygon.

    Parameters
    ----------
    blocks_gdf : geopandas.GeoDataFrame
        GeoDataFrame complying with :class:`BlocksSchema` whose geometry column
        contains polygons.

    Returns
    -------
    geopandas.GeoDataFrame
        Copy of the validated dataframe with an added ``centroid`` geometry
        column projected to the same CRS.

    Raises
    ------
    ValueError
        If validation fails for ``blocks_gdf``.
    """
    blocks_gdf = BlocksSchema(blocks_gdf)
    representative_point = blocks_gdf.representative_point()
    blocks_gdf[X_COLUMN] = representative_point.x
    blocks_gdf[Y_COLUMN] = representative_point.y
    return blocks_gdf
