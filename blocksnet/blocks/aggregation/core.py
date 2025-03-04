import geopandas as gpd
import pandas as pd
from loguru import logger
from tqdm import tqdm
from .schemas import BlocksSchema
from ...common.config import log_config

COUNT_COLUMN = "count"
OBJECT_INDEX_COLUMN = "_object_index"
BLOCK_ID_COLUMN = "_block_id"
INTERSECTION_AREA_COLUMN = "_intersection_area"


def _validate_input(blocks_gdf: gpd.GeoDataFrame, objects_gdf: gpd.GeoDataFrame):
    logger.info("Validating input.")
    if not isinstance(objects_gdf, gpd.GeoDataFrame):
        raise ValueError("objects_gdf must be an instance of GeoDataFrame.")
    if blocks_gdf.crs != objects_gdf.crs:
        logger.warning("CRS of objects_gdf and blocks_gdf do not match. Reprojecting. ")
        objects_gdf.to_crs(blocks_gdf.crs, inplace=True)
    if COUNT_COLUMN in objects_gdf.columns:
        logger.warning(
            f"Column {COUNT_COLUMN} found in objects_gdf. It will be taken into account and might affect the result."
        )
    else:
        objects_gdf[COUNT_COLUMN] = 1


def _intersect_objects_with_blocks(objects_gdf: gpd.GeoDataFrame, blocks_gdf: gpd.GeoDataFrame):
    objects_gdf = objects_gdf.copy()
    intersections_gdf = gpd.overlay(objects_gdf, blocks_gdf, how="intersection")
    intersections_gdf[INTERSECTION_AREA_COLUMN] = intersections_gdf.geometry.area
    return intersections_gdf


def _keep_largest_intersections(intersections_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    intersections_gdf = intersections_gdf.copy()
    idx = intersections_gdf.groupby(OBJECT_INDEX_COLUMN)[INTERSECTION_AREA_COLUMN].idxmax()
    return intersections_gdf.loc[idx]


def _label_intersections_with_blocks(intersections_gdf: gpd.GeoDataFrame, blocks_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    intersections_gdf = intersections_gdf.copy()
    intersections_gdf.geometry = intersections_gdf.representative_point()
    intersections_gdf = gpd.sjoin(intersections_gdf, blocks_gdf, predicate="intersects")
    return intersections_gdf.drop(columns=["geometry"]).rename({"index_right": "_block_id"}, axis=1)


def _get_agg_dict(objects_gdf: gpd.GeoDataFrame):
    agg_dict = {COUNT_COLUMN: "sum"}
    for col in objects_gdf.columns:
        if col == "geometry" or col in [OBJECT_INDEX_COLUMN, INTERSECTION_AREA_COLUMN, BLOCK_ID_COLUMN]:
            continue
        dtype = objects_gdf[col].dtype
        if pd.api.types.is_numeric_dtype(dtype):
            agg_dict[col] = "sum"
    return agg_dict


def _aggregate_objects(
    blocks_gdf: gpd.GeoDataFrame,
    objects_gdf: gpd.GeoDataFrame,
):
    blocks_gdf = blocks_gdf.copy()
    objects_gdf = objects_gdf.copy()
    objects_gdf[OBJECT_INDEX_COLUMN] = objects_gdf.index
    intersections_gdf = _intersect_objects_with_blocks(objects_gdf, blocks_gdf)
    intersections_gdf = _keep_largest_intersections(intersections_gdf)
    intersections_gdf = _label_intersections_with_blocks(intersections_gdf, blocks_gdf)

    intersections_groupby = intersections_gdf.groupby(BLOCK_ID_COLUMN)
    aggregated_blocks_gdf = intersections_groupby.agg(_get_agg_dict(objects_gdf))

    for col in aggregated_blocks_gdf.columns:
        if not col in blocks_gdf.columns:
            blocks_gdf[col] = 0
        blocks_gdf[col] = blocks_gdf[col].add(aggregated_blocks_gdf[col], fill_value=0)

    return blocks_gdf


def aggregate_objects(blocks_gdf: gpd.GeoDataFrame, objects_gdf: gpd.GeoDataFrame):
    blocks_gdf = BlocksSchema(blocks_gdf)
    objects_gdf = objects_gdf.copy()

    _validate_input(blocks_gdf, objects_gdf)

    # blocks_gdf = _initialize_columns(blocks_gdf, objects_gdf)

    logger.info("Aggregating objects.")
    geom_types = objects_gdf.geom_type.unique()
    for geom_type in tqdm(geom_types, disable=log_config.disable_tqdm):
        objects_gdf = objects_gdf[objects_gdf.geom_type == geom_type]
        blocks_gdf = _aggregate_objects(blocks_gdf, objects_gdf)

    return blocks_gdf
