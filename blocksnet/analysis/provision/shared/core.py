from functools import wraps
import pandas as pd
from loguru import logger
from tqdm import tqdm
import numpy as np
from .schemas import BlocksSchema
from ....utils import validation
from ....config import log_config

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
        validation.validate_accessibility_matrix(accessibility_matrix, blocks_df)
        blocks_df = BlocksSchema(blocks_df)
        return func(blocks_df, accessibility_matrix, *args, **kwargs)

    return wrapper


# def _provision_total(blocks_df: pd.DataFrame):
#     return blocks_df[DEMAND_WITHIN_COLUMN].sum() / blocks_df[DEMAND_COLUMN].sum()


@_validate_and_preprocess_input
def shared_provision(
    blocks_df: pd.DataFrame,
    accessibility_matrix: pd.DataFrame,
    accessibility: int,
) -> tuple[pd.DataFrame, float, float]:

    blocks_df = _initialize_provision_df(blocks_df)

    # Булева маска доступности
    accessibility_mask = accessibility_matrix <= accessibility

    # Подсчет населения в зоне доступности
    blocks_df[POPULATION_WITHIN_COLUMN] = accessibility_mask.T.mul(blocks_df["population"], axis=0).sum()

    # Подсчет доступной емкости
    blocks_df[CAPACITY_WITHIN_COLUMN] = accessibility_mask.mul(blocks_df["capacity"], axis=0).sum(axis=1)

    # Вычисление нагрузки (только где capacity > 0)
    blocks_df[LOAD_COLUMN] = (
        blocks_df[POPULATION_WITHIN_COLUMN].where(blocks_df["capacity"] > 0) / blocks_df["capacity"]
    )

    # Вычисление обеспеченности (только где population > 0)
    blocks_df[PROVISION_COLUMN] = (
        blocks_df[CAPACITY_WITHIN_COLUMN].where(blocks_df["population"] > 0) / blocks_df["population"]
    )

    logger.success("Provision assessment finished")

    return blocks_df
