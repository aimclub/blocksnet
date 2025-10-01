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

SHARE_LIVING_COLUMN = "share_living"
SHARE_NON_LIVING_COLUMN = "share_non_living"


def calculate_density_indicators(blocks_df: pd.DataFrame) -> pd.DataFrame:
    blocks_df = BlocksSchema(blocks_df)

    blocks_df = blocks_df.assign(
        **{
            FSI_COLUMN: blocks_df.build_floor_area / blocks_df.site_area,
            GSI_COLUMN: (blocks_df.footprint_area / blocks_df.site_area).clip(0, 1),
            MXI_COLUMN: (blocks_df.living_area / blocks_df.build_floor_area).clip(0, 1),
            L_COLUMN: (blocks_df.build_floor_area / blocks_df.footprint_area).clip(1),
            OSR_COLUMN: (blocks_df.site_area - blocks_df.footprint_area) / blocks_df.build_floor_area,
            SHARE_LIVING_COLUMN: blocks_df.living_area / blocks_df.footprint_area,
            SHARE_NON_LIVING_COLUMN: blocks_df.non_living_area / blocks_df.footprint_area,
        }
    )

    return blocks_df
