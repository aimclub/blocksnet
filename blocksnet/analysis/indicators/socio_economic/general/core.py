import pandas as pd
from .schemas import BlocksSchema
from .indicator import GeneralIndicator
from blocksnet.enums import LandUseCategory

SQM_IN_SQKM = 1000 * 1000


def calculate_general_indicators(blocks_df: pd.DataFrame) -> dict[GeneralIndicator, float]:
    blocks_df = BlocksSchema(blocks_df)

    area = blocks_df["site_area"].sum()
    blocks_df["category"] = blocks_df["land_use"].apply(LandUseCategory.from_land_use)
    urban_area = blocks_df[blocks_df["category"] == LandUseCategory.URBAN]["site_area"].sum()
    urbanization = urban_area / area

    area = area / SQM_IN_SQKM

    return {GeneralIndicator.AREA: float(area), GeneralIndicator.URBANIZATION: float(urbanization)}
