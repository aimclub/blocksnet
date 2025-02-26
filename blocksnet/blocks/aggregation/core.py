import geopandas as gpd
from loguru import logger
from .schemas import BlocksSchema, ObjectsSchema
from ...config import log_config


def aggregate(blocks_gdf: gpd.GeoDataFrame, objects_gdf: gpd.GeoDataFrame, split: bool = False):
    blocks_gdf = BlocksSchema(blocks_gdf)
    objects_gdf = BlocksSchema(objects_gdf)
