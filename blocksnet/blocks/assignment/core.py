import geopandas as gpd
import numpy as np
import pandas as pd
from loguru import logger
from .schemas import BlocksSchema, FunctionalZonesSchema
from .utils import sjoin_intersections
from blocksnet.enums import LandUse
from blocksnet.utils.validation import ensure_crs

LAND_USE_COLUMN = "land_use"
SHARE_COLUMN = "share"


def _get_shares(intersections_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    shares_df = intersections_gdf.groupby(["index_left", LAND_USE_COLUMN]).agg({"share_left": "sum"})
    shares_df = shares_df.unstack(LAND_USE_COLUMN, fill_value=0).droplevel(0, axis=1)
    return shares_df


def _choose_largest(shares_df: pd.DataFrame) -> pd.DataFrame:
    shares_df = shares_df.copy()
    lus = shares_df.idxmax(axis=1)
    shares = shares_df.max(axis=1)
    shares_df[LAND_USE_COLUMN] = lus.apply(lambda lu: LandUse(lu))
    shares_df[SHARE_COLUMN] = shares
    return shares_df[[LAND_USE_COLUMN, SHARE_COLUMN]]


def assign_land_use(
    blocks_gdf: gpd.GeoDataFrame,
    functional_zones_gdf: gpd.GeoDataFrame,
    rules: dict[str, LandUse],
):
    """Assign dominant land-use types to blocks based on functional zones.

    Parameters
    ----------
    blocks_gdf : geopandas.GeoDataFrame
        Block geometries that receive land-use labels.
    functional_zones_gdf : geopandas.GeoDataFrame
        Functional zoning polygons containing ``functional_zone`` names.
    rules : dict[str, LandUse]
        Mapping from functional zone names to :class:`LandUse` categories.

    Returns
    -------
    geopandas.GeoDataFrame
        Blocks GeoDataFrame augmented with land-use shares and dominant land
        use plus share columns.
    """

    blocks_gdf = BlocksSchema(blocks_gdf)
    functional_zones_gdf = FunctionalZonesSchema(functional_zones_gdf)
    ensure_crs(blocks_gdf, functional_zones_gdf)

    functional_zones_gdf[LAND_USE_COLUMN] = functional_zones_gdf.functional_zone.apply(
        lambda fz: rules[fz].value if fz in rules else None
    )
    functional_zones_gdf = functional_zones_gdf[~functional_zones_gdf[LAND_USE_COLUMN].isna()]

    logger.info("Overlaying geometries")
    intersections_gdf = sjoin_intersections(blocks_gdf, functional_zones_gdf)

    blocks_gdf[[lu.value for lu in LandUse]] = 0.0
    shares_df = _get_shares(intersections_gdf)
    blocks_gdf.loc[shares_df.index, shares_df.columns] = shares_df

    blocks_gdf[LAND_USE_COLUMN] = None
    blocks_gdf[SHARE_COLUMN] = np.nan
    blocks_gdf.loc[shares_df.index, [LAND_USE_COLUMN, SHARE_COLUMN]] = _choose_largest(shares_df)

    logger.success("Shares calculated")
    return blocks_gdf
