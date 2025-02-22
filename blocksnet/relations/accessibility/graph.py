from typing import Literal
import geopandas as gpd
import iduedu as ie
from loguru import logger
from .schemas import TerritorySchema
from .const import IDUEDU_CRS


def get_accessibility_graph(
    territory_gdf: gpd.GeoDataFrame, graph_type: Literal["drive", "walk", "intermodal"], *args, **kwargs
):
    territory_gdf = TerritorySchema(territory_gdf)
    if territory_gdf.crs.to_epsg() != IDUEDU_CRS:
        logger.info("CRS do not match IDUEDU required crs. Reprojecting.")
        territory_gdf = territory_gdf.to_crs(IDUEDU_CRS)
    geometry = territory_gdf.unary_union
    if graph_type == "drive":
        return ie.get_drive_graph(polygon=geometry, *args, **kwargs)
    if graph_type == "walk":
        return ie.get_walk_graph(polygon=geometry, *args, **kwargs)
    if graph_type == "intermodal":
        return ie.get_intermodal_graph(polygon=geometry, *args, **kwargs)
    raise ValueError("Graph type must be drive, walk or intermodal.")
