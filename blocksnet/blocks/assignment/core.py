import geopandas as gpd
import pandas as pd
from loguru import logger
from .schemas import BlocksSchema, FunctionalZonesSchema
from ...common.enums import LandUse
from ...common.spatial import sjoin_intersections
from ...common.validation import ensure_crs

FZ_SHARES_COLUMN = "fz_shares"
FZ_SHARE_COLUMN = "fz_share"
LU_SHARES_COLUMN = "lu_shares"
LU_SHARE_COLUMN = "lu_share"
LAND_USE_COLUMN = "land_use"


def _get_shares(intersections_gdf: gpd.GeoDataFrame, column: str) -> pd.Series:
    grouped = intersections_gdf.groupby(["index_left", column]).agg({"share_left": "sum"}).reset_index()
    series = grouped.groupby("index_left").apply(
        lambda df: dict(zip(df[column], df["share_left"])), include_groups=False
    )
    return series


def assign_land_use(
    blocks_gdf: gpd.GeoDataFrame,
    functional_zones_gdf: gpd.GeoDataFrame,
    rules: dict[str, LandUse],
):

    blocks_gdf = BlocksSchema(blocks_gdf)
    functional_zones_gdf = FunctionalZonesSchema(functional_zones_gdf)
    ensure_crs(blocks_gdf, functional_zones_gdf)

    functional_zones_gdf[LAND_USE_COLUMN] = functional_zones_gdf.functional_zone.apply(
        lambda fz: rules[fz].value if fz in rules else None
    )

    logger.info("Overlaying geometries.")
    intersections_gdf = sjoin_intersections(blocks_gdf, functional_zones_gdf)

    logger.info("Calculating shares.")
    blocks_gdf[FZ_SHARES_COLUMN] = _get_shares(intersections_gdf, "functional_zone")
    blocks_gdf[FZ_SHARE_COLUMN] = blocks_gdf[FZ_SHARES_COLUMN].apply(
        lambda s: max(s.values()) if isinstance(s, dict) else None
    )
    blocks_gdf["functional_zone"] = blocks_gdf[FZ_SHARES_COLUMN].apply(
        lambda s: max(s, key=s.get) if isinstance(s, dict) else None
    )

    blocks_gdf[LU_SHARES_COLUMN] = _get_shares(intersections_gdf, LAND_USE_COLUMN)
    blocks_gdf[LU_SHARE_COLUMN] = blocks_gdf[LU_SHARES_COLUMN].apply(
        lambda s: max(s.values()) if isinstance(s, dict) else None
    )
    blocks_gdf[LAND_USE_COLUMN] = blocks_gdf[LU_SHARES_COLUMN].apply(
        lambda s: max(s, key=s.get) if isinstance(s, dict) else None
    )

    logger.success("Shares calculated.")
    return blocks_gdf
