import pandas as pd
from .schemas import BlocksSchema
from .indicator import DemographicIndicator

SQM_IN_SQKM = 1000 * 1000


def calculate_demographic_indicators(blocks_df: pd.DataFrame) -> dict[DemographicIndicator, float]:
    blocks_df = BlocksSchema(blocks_df)

    area = blocks_df["site_area"].sum() / SQM_IN_SQKM
    population = blocks_df["population"].sum()
    density = population / area

    return {DemographicIndicator.POPULATION: int(population), DemographicIndicator.DENSITY: float(density)}
