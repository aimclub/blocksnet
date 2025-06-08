import pandas as pd
import geopandas as gpd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from loguru import logger
from blocksnet.enums import LandUse
from blocksnet.relations import validate_accessibility_matrix
from blocksnet.analysis.provision.diversity.core import shannon_diversity, SHANNON_DIVERSITY_COLUMN, COUNT_COLUMN
from blocksnet.utils.validation import ensure_crs
from .schemas import BlocksSchema, NodesSchema

DENSITY_COLUMN = "density"
LU_CONST_COLUMN = "lu_const"
ATTRACTIVENESS_COLUMN = "attractiveness"
POPULATION_COLUMN = "population"

LU_CONSTS = {
    LandUse.INDUSTRIAL: 0.25,
    LandUse.BUSINESS: 0.3,
    LandUse.SPECIAL: 0.1,
    LandUse.TRANSPORT: 0.1,
    LandUse.RESIDENTIAL: 0.1,
    LandUse.AGRICULTURE: 0.05,
    LandUse.RECREATION: 0.05,
}
DEFAULT_LU_CONST = 0.06
DEFAULT_ACCESSIBILITY = 10


def _calculate_nodes_weights(
    blocks_df: gpd.GeoDataFrame, nodes_gdf: gpd.GeoDataFrame, acc_mx: pd.DataFrame, accessibility: float
) -> None:

    logger.info("Identifying nearest nodes to blocks")
    acc_mx = acc_mx.replace(0, 0.1)
    acc_mask = acc_mx <= accessibility
    acc_mask = acc_mask | acc_mx.eq(acc_mx.min(axis=1), axis=0)

    logger.info("Calculating weights")
    weights_mx = pd.DataFrame(0.0, index=acc_mx.index, columns=acc_mx.columns)
    weights_mx[acc_mask] = 1.0 / acc_mx[acc_mask]
    weights_sum = weights_mx.sum(axis=1)
    weights_mx = weights_mx.div(weights_sum, axis=0)

    logger.info("Distributing")
    nodes_gdf[ATTRACTIVENESS_COLUMN] = weights_mx.mul(blocks_df[ATTRACTIVENESS_COLUMN], axis=0).sum(axis=0)
    nodes_gdf[POPULATION_COLUMN] = weights_mx.mul(blocks_df[POPULATION_COLUMN], axis=0).sum(axis=0)
    return nodes_gdf


def _calculate_diversity(blocks_df: pd.DataFrame, services_dfs: list[pd.DataFrame]) -> pd.DataFrame:
    logger.info("Calculating diversity and density")
    diversity_df = shannon_diversity(services_dfs)
    blocks_df = blocks_df.join(diversity_df)
    blocks_df[DENSITY_COLUMN] = blocks_df[COUNT_COLUMN] / blocks_df.site_area
    return blocks_df


def _calculate_attractiveness(blocks_df: pd.DataFrame, lu_consts: dict[LandUse, float]) -> pd.DataFrame:
    logger.info("Calculating attractiveness")
    blocks_df = blocks_df.copy()
    scaler = MinMaxScaler()
    columns = [DENSITY_COLUMN, SHANNON_DIVERSITY_COLUMN]
    blocks_df[columns] = scaler.fit_transform(blocks_df[columns])
    blocks_df[LU_CONST_COLUMN] = blocks_df.land_use.apply(lambda lu: lu_consts.get(lu, DEFAULT_LU_CONST))
    blocks_df[ATTRACTIVENESS_COLUMN] = (
        blocks_df[DENSITY_COLUMN] + blocks_df[SHANNON_DIVERSITY_COLUMN] + blocks_df[LU_CONST_COLUMN]
    )
    return blocks_df


def _calculate_od_mx(nodes_gdf: gpd.GeoDataFrame, acc_mx: pd.DataFrame) -> pd.DataFrame:
    logger.info("Calculating origin destination matrix")
    acc_mx = acc_mx.replace(0, np.nan)
    return pd.DataFrame(
        np.outer(nodes_gdf[POPULATION_COLUMN], nodes_gdf[ATTRACTIVENESS_COLUMN]) / acc_mx,
        index=acc_mx.index,
        columns=acc_mx.columns,
    ).fillna(0.0)


def origin_destination_matrix(
    blocks_df: pd.DataFrame,
    nodes_gdf: gpd.GeoDataFrame,
    blocks_to_nodes_mx: pd.DataFrame,
    nodes_to_nodes_mx: pd.DataFrame,
    services_dfs: list[pd.DataFrame],
    accessibility: float = DEFAULT_ACCESSIBILITY,
    lu_consts: dict[LandUse, float] = LU_CONSTS,
) -> pd.DataFrame:

    logger.info("Validating input data")
    blocks_df = BlocksSchema(blocks_df)
    # TODO validate_accessibility_matrix(blocks_to_nodes_acc_mx, blocks_gdf)
    nodes_gdf = NodesSchema(nodes_gdf)
    # TODO validate_accessibility_matrix(nodes_to_nodes_acc_mx, nodes_gdf)

    blocks_df = _calculate_diversity(blocks_df, services_dfs)
    blocks_df = _calculate_attractiveness(blocks_df, lu_consts)

    scaler = MinMaxScaler()
    blocks_df[[POPULATION_COLUMN, ATTRACTIVENESS_COLUMN]] = scaler.fit_transform(
        blocks_df[[POPULATION_COLUMN, ATTRACTIVENESS_COLUMN]]
    )

    nodes_gdf = _calculate_nodes_weights(blocks_df, nodes_gdf, blocks_to_nodes_mx, accessibility)

    return _calculate_od_mx(nodes_gdf, nodes_to_nodes_mx)
