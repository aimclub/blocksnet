import pandas as pd
from .schemas import BlocksSchema
from .indicator import DemographicIndicator
from ..const import SQM_IN_SQKM


def calculate_demographic_indicators(blocks_df: pd.DataFrame) -> dict[DemographicIndicator, float]:
    """Calculate population totals and densities for the study area.

    Parameters
    ----------
    blocks_df : pandas.DataFrame
        Block dataset validated by :class:`BlocksSchema`.

    Returns
    -------
    dict[DemographicIndicator, float]
        Population count and density keyed by indicator.
    """
    blocks_df = BlocksSchema(blocks_df)

    area = blocks_df["site_area"].sum() / SQM_IN_SQKM
    population = blocks_df["population"].sum()
    density = population / area

    return {DemographicIndicator.POPULATION: int(population), DemographicIndicator.DENSITY: float(density)}
