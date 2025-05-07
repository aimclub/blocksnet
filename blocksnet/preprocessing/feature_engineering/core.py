import geopandas as gpd
import shapely
from loguru import logger
from ...config import log_config
from .schemas import BlocksSchema
from . import utils

X_COLUMN = "x"
Y_COLUMN = "y"
AREA_COLUMN = "area"
LENGTH_COLUMN = "length"
CORNERS_COUNT_COLUMN = "corners_count"
OUTER_RADIUS_COLUMN = "outer_radius"
INNER_RADIUS_COLUMN = "inner_radius"
CENTERLINE_LENGTH_COLUMN = "centerline_length"
ASPECT_RATIO_COLUMN = "aspect_ratio"


def _calculate_centerlines(blocks_gdf: gpd.GeoDataFrame):
    logger.info("Calculating centerlines")
    try:
        import pygeoops
    except ImportError:
        raise ImportError("PyGeoOps package is required but not installed")
    blocks_gdf = blocks_gdf.copy()
    if log_config.disable_tqdm:
        blocks_gdf[CENTERLINE_LENGTH_COLUMN] = blocks_gdf.geometry.apply(pygeoops.centerline).length
    else:
        blocks_gdf[CENTERLINE_LENGTH_COLUMN] = blocks_gdf.geometry.progress_apply(pygeoops.centerline).length
    return blocks_gdf


def _calculate_usual_features(blocks_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    logger.info("Calculating usual features")
    blocks_gdf = blocks_gdf.copy()
    blocks_gdf[X_COLUMN] = blocks_gdf.representative_point().x
    blocks_gdf[Y_COLUMN] = blocks_gdf.representative_point().y
    blocks_gdf[AREA_COLUMN] = blocks_gdf.area
    blocks_gdf[LENGTH_COLUMN] = blocks_gdf.length
    blocks_gdf[CORNERS_COUNT_COLUMN] = blocks_gdf.geometry.apply(lambda g: len(g.exterior.coords))
    return blocks_gdf


def _calculate_radiuses(blocks_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    logger.info("Calculating radiuses")
    blocks_gdf = blocks_gdf.copy()
    if log_config.disable_tqdm:
        blocks_gdf[OUTER_RADIUS_COLUMN] = blocks_gdf.geometry.apply(utils.calculate_outer_radius)
        blocks_gdf[INNER_RADIUS_COLUMN] = blocks_gdf.geometry.apply(utils.calculate_inner_radius)
    else:
        blocks_gdf[OUTER_RADIUS_COLUMN] = blocks_gdf.geometry.progress_apply(utils.calculate_outer_radius)
        blocks_gdf[INNER_RADIUS_COLUMN] = blocks_gdf.geometry.progress_apply(utils.calculate_inner_radius)
    return blocks_gdf


def _calculate_aspect_ratios(blocks_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    logger.info("Calculating aspect ratios")
    blocks_gdf = blocks_gdf.copy()
    if log_config.disable_tqdm:
        blocks_gdf[ASPECT_RATIO_COLUMN] = blocks_gdf.geometry.apply(utils.calculate_aspect_ratio)
    else:
        blocks_gdf[ASPECT_RATIO_COLUMN] = blocks_gdf.geometry.progress_apply(utils.calculate_aspect_ratio)
    return blocks_gdf


def _generate_combinations(blocks_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    logger.info("Generating combinations")
    try:
        import featuretools as ft
    except ImportError:
        raise ImportError("Featuretools package is required but not installed")

    blocks_df = blocks_gdf.drop(columns=["geometry"]).copy()
    blocks_df["ft_index"] = blocks_df.index
    es = ft.EntitySet(id="")
    es = es.add_dataframe(dataframe_name="", dataframe=blocks_df, index="ft_index")
    df, _ = ft.dfs(
        entityset=es, target_dataframe_name="", max_depth=1, trans_primitives=["multiply_numeric", "divide_numeric"]
    )
    return df


def generate_geometries_features(
    blocks_gdf: gpd.GeoDataFrame,
    radiuses: bool = False,
    aspect_ratios: bool = False,
    centerlines: bool = False,
    combinations: bool = False,
):

    blocks_gdf = BlocksSchema(blocks_gdf)

    blocks_gdf = _calculate_usual_features(blocks_gdf)

    if radiuses:
        blocks_gdf = _calculate_radiuses(blocks_gdf)
    if aspect_ratios:
        blocks_gdf = _calculate_aspect_ratios(blocks_gdf)
    if centerlines:
        blocks_gdf = _calculate_centerlines(blocks_gdf)

    if combinations:
        combinations_df = _generate_combinations(blocks_gdf)
        blocks_gdf = blocks_gdf[["geometry"]].join(combinations_df)

    return blocks_gdf
