from functools import wraps
import pandas as pd
from loguru import logger
from tqdm import tqdm
import numpy as np
from .schemas import BlocksSchema
from blocksnet.relations import validate_accessibility_matrix
from blocksnet.config import log_config

CAPACITY_WITHIN_COLUMN = "capacity_within"
POPULATION_WITHIN_COLUMN = "population_within"
LOAD_COLUMN = "load"
PROVISION_COLUMN = "provision"


def _initialize_provision_df(blocks_df: pd.DataFrame):
    logger.info("Initializing provision DataFrame")
    blocks_df[CAPACITY_WITHIN_COLUMN] = 0
    blocks_df[POPULATION_WITHIN_COLUMN] = 0
    return blocks_df


def _validate_and_preprocess_input(func):
    @wraps(func)
    def wrapper(blocks_df: pd.DataFrame, accessibility_matrix: pd.DataFrame, *args, **kwargs):
        validate_accessibility_matrix(accessibility_matrix, blocks_df)
        blocks_df = BlocksSchema(blocks_df)
        return func(blocks_df, accessibility_matrix, *args, **kwargs)

    return wrapper


@_validate_and_preprocess_input
def shared_provision(
    blocks_df: pd.DataFrame,
    accessibility_matrix: pd.DataFrame,
    accessibility: int,
) -> pd.DataFrame:
    """Estimate service provision using shared accessibility buffers.

    Parameters
    ----------
    blocks_df : pandas.DataFrame
        Block dataframe containing population and capacity columns.
    accessibility_matrix : pandas.DataFrame
        Accessibility matrix whose units match ``accessibility``.
    accessibility : int
        Threshold accessibility cost used to define neighbourhoods.

    Returns
    -------
    pandas.DataFrame
        Dataframe with load and provision metrics per block.
    """

    blocks_df = _initialize_provision_df(blocks_df)

    accessibility_mask = accessibility_matrix <= accessibility

    blocks_df[POPULATION_WITHIN_COLUMN] = accessibility_mask.T.mul(blocks_df["population"], axis=0).sum()

    blocks_df[CAPACITY_WITHIN_COLUMN] = accessibility_mask.mul(blocks_df["capacity"], axis=0).sum(axis=1)

    blocks_df[LOAD_COLUMN] = (
        blocks_df[POPULATION_WITHIN_COLUMN].where(blocks_df["capacity"] > 0) / blocks_df["capacity"]
    )

    blocks_df[PROVISION_COLUMN] = (
        blocks_df[CAPACITY_WITHIN_COLUMN].where(blocks_df["population"] > 0) / blocks_df["population"]
    )

    logger.success("Provision assessment finished")

    return blocks_df
