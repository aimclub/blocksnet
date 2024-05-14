import geopandas as gpd
import pandas as pd
from pydantic import BaseModel, field_validator, model_validator
import shapely
from ..models.land_use import LandUse
from ..models.geodataframe import GeoDataFrame, BaseRow

LAND_USE_COLUMN = "land_use"


class BlockRow(BaseRow):
    geometry: shapely.Polygon


class ZoneRow(BaseRow):
    geometry: shapely.Polygon | shapely.MultiPolygon
    zone: str


class LandUseProcessor(BaseModel):
    blocks: GeoDataFrame[BlockRow]
    zones: GeoDataFrame[ZoneRow]
    zone_to_land_use: dict[str, LandUse] = {
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

    @field_validator("blocks", mode="before")
    @staticmethod
    def validate_blocks(blocks):
        if not isinstance(blocks, GeoDataFrame[BlockRow]):
            blocks = GeoDataFrame[BlockRow](blocks)
        return blocks

    @field_validator("zones", mode="before")
    @staticmethod
    def validate_zones(zones):
        if not isinstance(zones, GeoDataFrame[ZoneRow]):
            zones = GeoDataFrame[ZoneRow](zones)
        return zones

    @model_validator(mode="after")
    @staticmethod
    def validate_model(self):
        blocks = self.blocks
        zones = self.zones
        assert blocks.crs == zones.crs, "Blocks CRS must match zones CRS"
        assert zones.zone.isin(self.zone_to_land_use.keys()).all(), "Zone must be in dict"
        return self

    def get_blocks(self, min_intersection: float = 0.3) -> gpd.GeoDataFrame:
        # min_intersection : float – threshold, minimum area (share) covered by geometry from df_with_attribute. If intersection > min_intersection, then attribute is recorded to df

        # returns gpd.GeoDataFrame

        zones = self.zones.copy()
        zones[LAND_USE_COLUMN] = zones["zone"].replace(self.zone_to_land_use).apply(lambda x: x.name)

        block_id_column = "block_id"
        blocks = self.blocks.copy()
        blocks[block_id_column] = blocks.index

        # overlay geometries
        df_temp = gpd.overlay(
            blocks[[block_id_column, "geometry"]],
            zones[[LAND_USE_COLUMN, "geometry"]],
            how="intersection",
            keep_geom_type=False,
        )

        # get areas
        df_temp["intersection_area"] = df_temp.area
        blocks["area"] = blocks.area

        # group by id and attribute value and find a sum of intersection areas
        df_temp = df_temp.groupby([block_id_column, LAND_USE_COLUMN])["intersection_area"].sum().reset_index()

        df_temp = df_temp.merge(blocks[[block_id_column, "area"]], how="left")

        # find intersection shares
        df_temp["intersection_area"] = df_temp["intersection_area"] / df_temp["area"]

        # filter out rows with attribute lower than min_intersection
        df_temp = df_temp[df_temp["intersection_area"] > min_intersection]

        # get intersecting attributes for each id
        res = df_temp.groupby(block_id_column)[LAND_USE_COLUMN].apply(list).apply(pd.Series).reset_index()

        # intersecting attributes are added to columns x_1, x_2, x_3..., sorted by intersection area in descending order
        res.columns = [block_id_column] + [LAND_USE_COLUMN + "_" + str(i + 1) for i in range(len(res.columns) - 1)]
        res = blocks.drop("area", axis=1).merge(res, how="left")

        return res.drop(columns=[block_id_column])
