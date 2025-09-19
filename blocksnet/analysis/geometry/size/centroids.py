import geopandas as gpd
from .schemas import BlocksSchema

X_COLUMN = "x"
Y_COLUMN = "y"


def calculate_centroids(blocks_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    blocks_gdf = BlocksSchema(blocks_gdf)
    representative_point = blocks_gdf.representative_point()
    blocks_gdf[X_COLUMN] = representative_point.x
    blocks_gdf[Y_COLUMN] = representative_point.y
    return blocks_gdf
