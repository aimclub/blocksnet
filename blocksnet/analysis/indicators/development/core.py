import pandas as pd
from .schemas import BlocksSchema

BUILD_FLOOR_AREA_COLUMN = "build_floor_area"
FOOTPRINT_AREA_COLUMN = "footprint_area"
LIVING_AREA_COLUMN = "living_area"
NON_LIVING_AREA_COLUMN = "non_living_area"


def calculate_development_indicators(blocks_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate development indicators.

    Parameters
    ----------
    blocks_df : pd.DataFrame
        Description.

    Returns
    -------
    pd.DataFrame
        Description.

    """
    blocks_df = BlocksSchema(blocks_df)

    build_floor_area = blocks_df.fsi * blocks_df.site_area
    footprint_area = blocks_df.gsi * blocks_df.site_area
    living_area = blocks_df.mxi * build_floor_area
    non_living_area = build_floor_area - living_area

    blocks_df = blocks_df.assign(
        **{
            BUILD_FLOOR_AREA_COLUMN: build_floor_area,
            FOOTPRINT_AREA_COLUMN: footprint_area,
            LIVING_AREA_COLUMN: living_area,
            NON_LIVING_AREA_COLUMN: non_living_area,
        }
    )

    return blocks_df
