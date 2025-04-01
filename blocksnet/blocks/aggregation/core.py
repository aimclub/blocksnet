import geopandas as gpd
import pandas as pd
from loguru import logger
from tqdm import tqdm
from .schemas import BlocksSchema
from ...utils.validation import ensure_crs

OBJECTS_COUNT_COLUMN = "objects_count"


def _preprocess_input(blocks_gdf: gpd.GeoDataFrame, objects_gdf: gpd.GeoDataFrame):
    logger.info("Preprocessing input")
    ensure_crs(blocks_gdf, objects_gdf)
    objects_gdf["geometry"] = objects_gdf.representative_point()
    if OBJECTS_COUNT_COLUMN in objects_gdf.columns:
        logger.warning(
            f"Column {OBJECTS_COUNT_COLUMN} found in objects_gdf. It will be taken into account and might affect the result"
        )
    else:
        objects_gdf[OBJECTS_COUNT_COLUMN] = 1


def _get_agg_rules(objects_gdf: gpd.GeoDataFrame):
    agg_dict = {}
    for col in objects_gdf.columns:
        if col == "geometry":
            continue
        dtype = objects_gdf[col].dtype
        if pd.api.types.is_numeric_dtype(dtype):
            agg_dict[col] = "sum"
    return agg_dict


def aggregate_objects(
    blocks_gdf: gpd.GeoDataFrame, objects_gdf: gpd.GeoDataFrame
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    blocks_gdf = BlocksSchema(blocks_gdf)
    objects_gdf = objects_gdf.copy()
    _preprocess_input(blocks_gdf, objects_gdf)

    logger.info("Aggregating objects")
    agg_rules = _get_agg_rules(objects_gdf)
    sjoin = objects_gdf.sjoin(blocks_gdf)
    agg = sjoin.groupby("index_right").agg(agg_rules)
    blocks_gdf = blocks_gdf.join(agg).fillna(0)

    excess_objects_gdf = objects_gdf[~objects_gdf.index.isin(sjoin.index)].copy()

    return blocks_gdf, excess_objects_gdf
