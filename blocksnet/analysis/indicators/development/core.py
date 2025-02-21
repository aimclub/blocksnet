import pandas as pd
import numpy as np
from loguru import logger
from functools import wraps
from .schemas import BlocksSchema

FSI_COLUMN = "fsi"
GSI_COLUMN = "gsi"
MXI_COLUMN = "mxi"
L_COLUMN = "l"
OSR_COLUMN = "osr"

LIVING_DEMAND_COLUMN = "living_demand"
SHARE_LIVING_COLUMN = "share_living"
SHARE_BUSINESS_COLUMN = "share_living"

LIVING_AREA_DEMAND = 20


def _restore_population(blocks_df: pd.DataFrame):
    if "population" not in blocks_df.columns:
        logger.info("population not in columns, restoring")
        if "living_area" in blocks_df.columns:
            blocks_df["population"] = blocks_df["living_area"] // LIVING_AREA_DEMAND


def _restore_business_area(blocks_df: pd.DataFrame):
    if "business_area" not in blocks_df.columns:
        logger.info("business_area not in columns, restoring")
        if "living_area" in blocks_df.columns and "build_floor_area" in blocks_df.columns:
            blocks_df["business_area"] = blocks_df["build_floor_area"] - blocks_df["living_area"]


def _restore_missing_columns(func):
    @wraps(func)
    def wrapper(blocks_df: pd.DataFrame, *args, **kwargs):
        blocks_df = blocks_df.copy()
        _restore_population(blocks_df)
        _restore_business_area(blocks_df)
        return func(blocks_df, *args, **kwargs)

    return wrapper


@_restore_missing_columns
def calculate_development_indicators(blocks_df: pd.DataFrame) -> pd.DataFrame:
    blocks_df = BlocksSchema(blocks_df)

    blocks_df = blocks_df.assign(
        **{
            FSI_COLUMN: blocks_df.build_floor_area / blocks_df.site_area,
            GSI_COLUMN: blocks_df.footprint_area / blocks_df.site_area,
            MXI_COLUMN: blocks_df.living_area / blocks_df.build_floor_area,
            L_COLUMN: blocks_df.build_floor_area / blocks_df.footprint_area,
            OSR_COLUMN: (blocks_df.site_area - blocks_df.footprint_area) / blocks_df.build_floor_area,
            SHARE_BUSINESS_COLUMN: blocks_df.business_area / blocks_df.footprint_area,
            SHARE_LIVING_COLUMN: blocks_df.living_area / blocks_df.footprint_area,
            LIVING_DEMAND_COLUMN: blocks_df.living_area / blocks_df.population,
        }
    )

    return blocks_df


__all__ = [
    "FSI_COLUMN",
    "GSI_COLUMN",
    "MXI_COLUMN",
    "L_COLUMN",
    "OSR_COLUMN",
    "LIVING_DEMAND_COLUMN",
    "SHARE_LIVING_COLUMN",
    "SHARE_BUSINESS_COLUMN",
    "calculate_development_indicators",
]
