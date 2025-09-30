import pandas as pd
from blocksnet.enums.land_use import LandUse
from blocksnet.config import land_use_config
from blocksnet.analysis.indicators.development.core import calculate_development_indicators, LIVING_AREA_COLUMN

SHARE_COLUMN = "share"

SITE_AREA_COLUMN = "site_area"

FSI_COLUMN = "fsi"
GSI_COLUMN = "gsi"
MXI_COLUMN = "mxi"
DEFAULT_MXI = 0.9

POPULATION_COLUMN = "population"
DEFAULT_LIVING_DEMAND = 30


def _calculate_site_area(shares_df: pd.DataFrame, area: float):
    """Calculate site area.

    Parameters
    ----------
    shares_df : pd.DataFrame
        Description.
    area : float
        Description.

    """
    shares_df = shares_df.copy()
    shares_df[SITE_AREA_COLUMN] = shares_df[SHARE_COLUMN] * area
    return shares_df


def _calculate_development(shares_df: pd.DataFrame, mxi: float):
    """Calculate development.

    Parameters
    ----------
    shares_df : pd.DataFrame
        Description.
    mxi : float
        Description.

    """
    density_df = shares_df.copy()
    density_df[FSI_COLUMN] = density_df.apply(lambda s: land_use_config.fsi_ranges.loc[s.name].mean(), axis=1)
    density_df[GSI_COLUMN] = density_df.apply(lambda s: land_use_config.gsi_ranges.loc[s.name].mean(), axis=1)
    density_df[MXI_COLUMN] = density_df.apply(lambda s: mxi if s.name == LandUse.RESIDENTIAL else 0, axis=1)

    development_df = calculate_development_indicators(density_df.reset_index())
    development_df.index = density_df.index
    new_columns = set(development_df.columns) & set(density_df.columns)
    return shares_df.join(development_df.drop(columns=new_columns))


def _calculate_population(shares_df: pd.DataFrame, living_demand: float):
    """Calculate population.

    Parameters
    ----------
    shares_df : pd.DataFrame
        Description.
    living_demand : float
        Description.

    """
    shares_df = shares_df.copy()
    shares_df[POPULATION_COLUMN] = (shares_df[LIVING_AREA_COLUMN] // living_demand).astype(int)
    return shares_df


def calculate_land_use_indicators(
    shares: dict[LandUse, float], area: float, mxi: float = DEFAULT_MXI, living_demand: float = DEFAULT_LIVING_DEMAND
):
    """Calculate land use indicators.

    Parameters
    ----------
    shares : dict[LandUse, float]
        Description.
    area : float
        Description.
    mxi : float, default: DEFAULT_MXI
        Description.
    living_demand : float, default: DEFAULT_LIVING_DEMAND
        Description.

    """
    shares_df = pd.DataFrame.from_dict(shares, orient="index", columns=[SHARE_COLUMN])
    shares_df = _calculate_site_area(shares_df, area)
    shares_df = _calculate_development(shares_df, mxi)
    shares_df = _calculate_population(shares_df, living_demand)
    return shares_df.drop(columns=[SHARE_COLUMN]).sum().to_dict()
