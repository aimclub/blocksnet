import geopandas as gpd
import shapely
from loguru import logger
from .schemas import BlocksSchema, FunctionalZonesSchema
from ...common.enums import LandUse

LAND_USE_COLUMN = "land_use"
SHARES_COLUMN = "shares"


def _ensure_crs(blocks_gdf: gpd.GeoDataFrame, functional_zones_gdf: gpd.GeoDataFrame):
    if blocks_gdf.crs != functional_zones_gdf.crs:
        logger.warning("CRS of functional_zones_gdf and blocks_gdf do not match. Reprojecting.")
        functional_zones_gdf.set_crs(blocks_gdf.crs, inplace=True)


def assign_land_use(
    blocks_gdf: gpd.GeoDataFrame,
    functional_zones_gdf: gpd.GeoDataFrame,
    rules: dict[str, LandUse],
    min_intersection_share: float = 0.3,
):

    blocks_gdf = BlocksSchema(blocks_gdf)
    functional_zones_gdf = FunctionalZonesSchema(functional_zones_gdf)

    _ensure_crs(blocks_gdf, functional_zones_gdf)

    logger.info("Intersecting geometries")

    sjoin_gdf = blocks_gdf.sjoin(functional_zones_gdf, predicate="intersects")

    def _get_shares(series):
        block_i = series.name
        block_geometry = series.geometry
        block_area = block_geometry.area
        gdf = sjoin_gdf[sjoin_gdf.index == block_i]
        shares = {}
        for zone_i in gdf["index_right"]:
            land_use = functional_zones_gdf.loc[zone_i, LAND_USE_COLUMN]
            zone_geometry = functional_zones_gdf.loc[zone_i, "geometry"]
            intersection_geometry = shapely.intersection(zone_geometry, block_geometry)
            intersection_area = intersection_geometry.area
            intersection_share = intersection_area / block_area
            if intersection_share >= min_intersection_share:
                shares[land_use] = intersection_area / block_area
        return shares

    logger.info("Calculating shares")

    blocks_gdf[SHARES_COLUMN] = blocks_gdf.apply(_get_shares, axis=1)
    blocks_gdf[LAND_USE_COLUMN] = blocks_gdf[SHARES_COLUMN].apply(
        lambda shares: max(shares, key=shares.get) if len(shares) > 0 else None
    )
    logger.success("Shares calculated")
    return blocks_gdf
