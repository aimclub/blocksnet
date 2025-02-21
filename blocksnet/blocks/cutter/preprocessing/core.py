import pandas as pd
import geopandas as gpd
import shapely
from loguru import logger
from .schemas import RoadsSchema, RailwaysSchema, WaterSchema
from functools import reduce, wraps


def _validate_gdfs(func):
    @wraps(func)
    def wrapper(
        roads_gdf: gpd.GeoDataFrame, railways_gdf: gpd.GeoDataFrame, water_gdf: gpd.GeoDataFrame, *args, **kwargs
    ):
        gdfs = [gdf for gdf in [roads_gdf, railways_gdf, water_gdf] if gdf is not None]
        if len(gdfs) == 0:
            raise ValueError("At least one GeoDataFrame must be passed.")

        all_crs = {gdf.crs for gdf in gdfs}
        if len(all_crs) > 1:
            raise ValueError(f"All GeoDataFrames must have equal crs")

        return func(roads_gdf, railways_gdf, water_gdf, *args, **kwargs)

    return wrapper


@_validate_gdfs
def preprocess_urban_objects(
    roads_gdf: gpd.GeoDataFrame | None = None,
    railways_gdf: gpd.GeoDataFrame | None = None,
    water_gdf: gpd.GeoDataFrame | None = None,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:

    crs = {gdf.crs for gdf in [roads_gdf, railways_gdf, water_gdf]}.pop()

    logger.info("Checking roads schema")
    if roads_gdf is None:
        roads_gdf = RoadsSchema.create_empty(crs)
    else:
        roads_gdf = RoadsSchema(roads_gdf)

    logger.info("Checking railways schema")
    if railways_gdf is None:
        railways_gdf = RailwaysSchema.create_empty(crs)
    else:
        railways_gdf = RailwaysSchema(railways_gdf)

    logger.info("Checking water schema")
    if water_gdf is None:
        water_gdf = WaterSchema.create_empty(crs)
    else:
        water_gdf = WaterSchema(water_gdf)

    polygon_objects = water_gdf[water_gdf.geometry.apply(lambda g: isinstance(g, shapely.Polygon))].copy()
    line_objects = pd.concat(
        [
            gdf[gdf.geometry.apply(lambda g: isinstance(g, shapely.LineString))]
            for gdf in [roads_gdf, railways_gdf, water_gdf]
        ]
    )
    return line_objects, polygon_objects
