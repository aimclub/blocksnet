import pandas as pd
from .schemas import BlocksAreaSchema
from ..count.core import services_count, COUNT_COLUMN

DENSITY_COLUMN = "density"


def services_density(blocks_df: pd.DataFrame):
    count_df = services_count(blocks_df)
    area_df = BlocksAreaSchema(blocks_df)
    blocks_df = pd.concat([area_df, count_df], axis=1)
    blocks_df[DENSITY_COLUMN] = blocks_df[COUNT_COLUMN] / blocks_df.site_area
    return blocks_df
