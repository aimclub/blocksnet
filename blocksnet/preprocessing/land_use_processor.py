"""
Module that processes data to classify blocks based on their intersection with predefined standardized zones.
"""
import geopandas as gpd
import pandas as pd
import pandera as pa
import shapely
from loguru import logger
from pandera.typing import Series
from ..models import BaseSchema
from ..models.land_use import LandUse

DEFAULT_ZONE_TO_LAND_USE = {
    "Т3Ж1": LandUse.RESIDENTIAL,
    "ТР0-2": LandUse.RECREATION,
    "Т3Ж2": LandUse.RESIDENTIAL,
    "Т1Ж2-1": LandUse.RESIDENTIAL,
    "Т2ЖД2": LandUse.RESIDENTIAL,
    "ТД1-3": LandUse.BUSINESS,
    "ТД2": LandUse.BUSINESS,
    "ТД3": LandUse.BUSINESS,
    "ТУ": LandUse.TRANSPORT,
    "ТИ4": LandUse.TRANSPORT,
    "ТД1-1": LandUse.RESIDENTIAL,
    "ТД1-2": LandUse.RESIDENTIAL,
    "ТПД1": LandUse.INDUSTRIAL,
    "ТПД2": LandUse.INDUSTRIAL,
    "ТИ1-1": LandUse.TRANSPORT,
    "Т3ЖД3": LandUse.RESIDENTIAL,
    "ТК1": LandUse.SPECIAL,
    "ТР2": LandUse.RECREATION,
    "ТИ2": LandUse.TRANSPORT,
    "ТР5-2": LandUse.RECREATION,
    "Т1Ж2-2": LandUse.RESIDENTIAL,
    "ТР4": LandUse.RECREATION,
    "ТР5-1": LandUse.RECREATION,
    "Т2Ж1": LandUse.RESIDENTIAL,
    "ТИ3": LandUse.TRANSPORT,
    "Т1Ж1": LandUse.RESIDENTIAL,
    "ТИ1-2": LandUse.TRANSPORT,
    "ТР3-2": LandUse.RECREATION,
    "ТР0-1": LandUse.RECREATION,
    "ТП2": LandUse.INDUSTRIAL,
    "ТК3": LandUse.SPECIAL,
    "ТР1": LandUse.RECREATION,
    "ТР3-1": LandUse.RECREATION,
    "ТС1": LandUse.AGRICULTURE,
    "ТК2": LandUse.SPECIAL,
    "ТП1": LandUse.INDUSTRIAL,
    "ТП3": LandUse.INDUSTRIAL,
    "ТП4": LandUse.INDUSTRIAL,
    "ТС2": LandUse.SPECIAL,
}

LAND_USE_COLUMN = "land_use"
ZONE_COLUMN = "zone"


class BlocksSchema(BaseSchema):
    """
    Schema for validating blocks GeoDataFrame.

    Attributes
    ----------
    _geom_types : list
        List of valid geometry types for the schema, set to shapely.Polygon.
    """

    _geom_types = [shapely.Polygon]


class ZonesSchema(BaseSchema):
    """
    Schema for validating zones GeoDataFrame.

    Attributes
    ----------
    _geom_types : list
        List of valid geometry types for the schema, set to shapely.Polygon and shapely.MultiPolygon.
    zone : Series[str]
        Pandera Series to enforce zone column type as string.
    """

    _geom_types = [shapely.Polygon, shapely.MultiPolygon]
    zone: Series[str]


class ProcessedBlocksSchema(BlocksSchema):
    """
    Extended BlocksSchema to include zone and land use information.

    Attributes
    ----------
    zone : Series[str]
        Pandera Series to enforce zone column type as string. Nullable, default is None.
    land_use : Series[str]
        Pandera Series to enforce land use column type as string. Nullable, default is None.
    """

    zone: Series[str] = pa.Field(nullable=True, default=None)
    land_use: Series[str] = pa.Field(nullable=True, default=None)


class LandUseProcessor:
    """
    Processes data to classify blocks based on their intersection with predefined standardized zones.

    Parameters
    ----------
    blocks : gpd.GeoDataFrame
        GeoDataFrame containing block data. Must contain the following columns:
        - index : int
        - geometry : Polygon

    zones : gpd.GeoDataFrame
        GeoDataFrame containing zone data. Must contain the following columns:
        - index : int
        - geometry : Polygon or MultiPolygon
        - zone : str

    zone_to_land_use : dict[str, LandUse], optional
        Dictionary mapping zone codes to LandUse enums, default is DEFAULT_ZONE_TO_LAND_USE.

    Raises
    ------
    AssertionError
        If the Coordinate Reference Systems (CRS) of `blocks` and `zones` do not match.
        If any zone in `zones` is not present in `zone_to_land_use` keys.

    Methods
    -------
    run(min_intersection: float = 0.3) -> gpd.GeoDataFrame
        Processes the blocks and zones to classify the blocks based on land use.

        Parameters
        ----------
        min_intersection : float
            Threshold, minimum area (share) covered by geometry from df_with_attribute. If intersection > min_intersection, then attribute is recorded to df, default is 0.3.

        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame containing the processed blocks with land use classifications.
    """

    def __init__(
        self,
        blocks: gpd.GeoDataFrame,
        zones: gpd.GeoDataFrame,
        zone_to_land_use: dict[str, LandUse] = DEFAULT_ZONE_TO_LAND_USE,
    ):

        blocks = BlocksSchema(blocks)
        zones = ZonesSchema(zones)

        assert blocks.crs == zones.crs, "Blocks CRS must match zones CRS"
        assert zones.zone.isin(zone_to_land_use.keys()).all(), "Zone must be in dict"

        self.blocks = blocks
        self.zones = zones
        self.zone_to_land_use = zone_to_land_use

    def run(self, min_intersection: float = 0.3) -> gpd.GeoDataFrame:
        """
        Processes the blocks and zones to classify the blocks based on land use.

        Parameters
        ----------
        min_intersection : float
            Threshold, minimum area (share) covered by geometry from df_with_attribute. If intersection > min_intersection, then attribute is recorded to df, default is 0.3.

        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame containing the processed blocks with land use classifications.
        """

        zones = self.zones.copy()
        # zones[LAND_USE_COLUMN] = zones[ZONE_COLUMN].replace(self.zone_to_land_use).apply(lambda x: x.name)

        block_id_column = "block_id"
        blocks = self.blocks.copy()
        blocks[block_id_column] = blocks.index

        # overlay geometries
        logger.info("Overlaying geometries")
        df_temp = gpd.overlay(
            blocks[[block_id_column, "geometry"]],
            zones[[ZONE_COLUMN, "geometry"]],
            how="intersection",
            keep_geom_type=False,
        )

        # get areas
        df_temp["intersection_area"] = df_temp.area
        blocks["area"] = blocks.area

        # group by id and attribute value and find a sum of intersection areas
        logger.info("Finding sum of intersection areas")
        df_temp = df_temp.groupby([block_id_column, ZONE_COLUMN])["intersection_area"].sum().reset_index()

        df_temp = df_temp.merge(blocks[[block_id_column, "area"]], how="left")

        # find intersection shares
        df_temp["intersection_area"] = df_temp["intersection_area"] / df_temp["area"]

        # filter out rows with attribute lower than min_intersection
        df_temp = df_temp[df_temp["intersection_area"] > min_intersection]

        # get intersecting attributes for each id
        logger.info("Getting intersecting attributes")
        res = df_temp.groupby(block_id_column)[ZONE_COLUMN].apply(list).apply(pd.Series).reset_index()

        # intersecting attributes are added to columns x_1, x_2, x_3..., sorted by intersection area in descending order
        res.columns = [block_id_column] + [ZONE_COLUMN + "_" + str(i + 1) for i in range(len(res.columns) - 1)]
        res = blocks.drop("area", axis=1).merge(res, how="left")
        res = res.rename(columns={f"{ZONE_COLUMN}_1": ZONE_COLUMN})
        res[LAND_USE_COLUMN] = (
            res[ZONE_COLUMN].replace(self.zone_to_land_use).apply(lambda x: x.name if isinstance(x, LandUse) else None)
        )

        return ProcessedBlocksSchema(res)
