import pandas as pd
import geopandas as gpd
import shapely
from loguru import logger
from .schemas import RoadsSchema, RailwaysSchema, WaterSchema
from functools import wraps
from blocksnet.utils.validation import ensure_crs


def _validate_gdfs(func):
    @wraps(func)
    def wrapper(
        roads_gdf: gpd.GeoDataFrame | None,
        railways_gdf: gpd.GeoDataFrame | None,
        water_gdf: gpd.GeoDataFrame | None,
        *args,
        **kwargs,
    ):

        roads_gdf = None if roads_gdf is None else roads_gdf.copy()
        railways_gdf = None if railways_gdf is None else railways_gdf.copy()
        water_gdf = None if water_gdf is None else water_gdf.copy()

        gdfs = [gdf for gdf in [roads_gdf, railways_gdf, water_gdf] if gdf is not None]
        if len(gdfs) == 0:
            raise ValueError("At least one GeoDataFrame must be passed")

        ensure_crs(*gdfs)

        return func(roads_gdf, railways_gdf, water_gdf, *args, **kwargs)

    return wrapper


@_validate_gdfs
def preprocess_urban_objects(
    roads_gdf: gpd.GeoDataFrame | None = None,
    railways_gdf: gpd.GeoDataFrame | None = None,
    water_gdf: gpd.GeoDataFrame | None = None,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:

    crs = {gdf.crs for gdf in [roads_gdf, railways_gdf, water_gdf] if gdf is not None}.pop()

    logger.info("Checking roads schema")
    if roads_gdf is None or roads_gdf.empty:
        logger.warning("Roads GeoDataFrame is None. Creating empty")
        roads_gdf = RoadsSchema.create_empty(crs)
    else:
        roads_gdf = RoadsSchema(roads_gdf)

    logger.info("Checking railways schema")
    if railways_gdf is None or railways_gdf.empty:
        logger.warning("Railways GeoDataFrame is None. Creating empty")
        railways_gdf = RailwaysSchema.create_empty(crs)
    else:
        railways_gdf = RailwaysSchema(railways_gdf)

    logger.info("Checking water schema")
    if water_gdf is None or water_gdf.empty:
        logger.warning("Water GeoDataFrame is None. Creating empty")
        water_gdf = WaterSchema.create_empty(crs)
    else:
        water_gdf = WaterSchema(water_gdf)

    ensure_crs(roads_gdf, railways_gdf, water_gdf)

    objects_gdf = pd.concat([roads_gdf, railways_gdf, water_gdf])
    polygons_gdf = objects_gdf[
        objects_gdf.geometry.apply(lambda g: isinstance(g, shapely.Polygon) or isinstance(g, shapely.MultiPolygon))
    ].reset_index(drop=True)
    lines_gdf = objects_gdf[
        objects_gdf.geometry.apply(
            lambda g: isinstance(g, shapely.LineString) or isinstance(g, shapely.MultiLineString)
        )
    ].reset_index(drop=True)

    return lines_gdf, polygons_gdf
