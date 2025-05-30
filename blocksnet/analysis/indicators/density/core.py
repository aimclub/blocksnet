from functools import wraps

import numpy as np
import pandas as pd
from loguru import logger

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
            GSI_COLUMN: blocks_df.footprint_area / blocks_df.site_area,
            MXI_COLUMN: blocks_df.living_area / blocks_df.build_floor_area,
            L_COLUMN: blocks_df.build_floor_area / blocks_df.footprint_area,
            OSR_COLUMN: (blocks_df.site_area - blocks_df.footprint_area) / blocks_df.build_floor_area,
            SHARE_LIVING_COLUMN: blocks_df.living_area / blocks_df.footprint_area,
            SHARE_NON_LIVING_COLUMN: blocks_df.non_living_area / blocks_df.footprint_area,
        }
    )

    return blocks_df
