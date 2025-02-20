import pandas as pd
import geopandas as gpd
import shapely
from ...utils.validation import GdfSchema
from pandera import dataframe_parser
from loguru import logger
from .schemas import BoundariesSchema, RoadsSchema, RailwaysSchema, WaterSchema


def preprocess_urban_objects(
    boundaries_gdf: gpd.GeoDataFrame,
    roads_gdf: gpd.GeoDataFrame | None,
    railways_gdf: gpd.GeoDataFrame | None,
    water_gdf: gpd.GeoDataFrame | None,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:

    logger.info("Checking boundaries schema")
    boundaries_gdf = BoundariesSchema(boundaries_gdf)
    crs = boundaries_gdf.crs

    logger.info("Checking roads schema")
    if roads_gdf is None:
        roads_gdf = RoadsSchema.create_empty().to_crs(crs)
    else:
        roads_gdf = RoadsSchema(roads_gdf)

    logger.info("Checking railways schema")
    if railways_gdf is None:
        railways_gdf = RailwaysSchema.create_empty().to_crs(crs)
    else:
        railways_gdf = RailwaysSchema(railways_gdf)

    logger.info("Checking water schema")
    if water_gdf is None:
        water_gdf = WaterSchema.create_empty().to_crs(crs)
    else:
        water_gdf = WaterSchema(water_gdf)

    for gdf in [roads_gdf, railways_gdf, water_gdf]:
        assert gdf.crs == crs, "CRS must match for all geodataframes"

    polygon_objects = water_gdf[water_gdf.geometry.apply(lambda g: isinstance(g, shapely.Polygon))].copy()
    line_objects = pd.concat(
        [
            gdf[gdf.geometry.apply(lambda g: isinstance(g, shapely.LineString))]
            for gdf in [roads_gdf, railways_gdf, water_gdf]
        ]
    )
    return boundaries_gdf, line_objects, polygon_objects
