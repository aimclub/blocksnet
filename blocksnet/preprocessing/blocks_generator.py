from typing import Any
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely import Polygon, LineString, MultiPolygon
from shapely.ops import polygonize
from pydantic import BaseModel, field_validator
from . import utils
from ..models import GeoDataFrame, BaseRow


class TerritoryRow(BaseRow):
    geometry: Polygon | MultiPolygon


class RoadsRow(BaseRow):
    geometry: LineString


class RailwaysRow(BaseRow):
    geometry: LineString


class WaterRow(BaseRow):
    geometry: LineString | Polygon | MultiPolygon


class BlockRow(BaseRow):
    geometry: Polygon


class BlocksGenerator(BaseModel):
    territory: GeoDataFrame[TerritoryRow]
    roads: GeoDataFrame[RoadsRow] = None
    railways: GeoDataFrame[RailwaysRow] = None
    water: GeoDataFrame[WaterRow] = None
    verbose: bool = True

    @field_validator("territory", mode="before")
    def validate_territory(value):
        return GeoDataFrame[TerritoryRow](value)

    @field_validator("roads", mode="before")
    def validate_roads(value):
        return GeoDataFrame[RoadsRow](value)

    @field_validator("railways", mode="before")
    def validate_railways(value):
        return GeoDataFrame[RailwaysRow](value)

    @field_validator("water", mode="before")
    def validate_water(value):
        return GeoDataFrame[WaterRow](value)

    @property
    def local_crs(self):
        return self.territory.crs

    def model_post_init(self, __context: Any) -> None:
        assert self.territory.crs == self.roads.crs, "Roads CRS have to match territory CRS"
        assert self.territory.crs == self.railways.crs, "Railways CRS have to match territory CRS"
        assert self.territory.crs == self.water.crs, "Water CRS have to match territory CRS"
        if self.water is not None:
            self.territory = self.territory.overlay(
                self.water[self.water.geom_type != "LineString"], how="difference"
            )  # cut water polygons from territory
            self.water = self.water["geometry"].apply(lambda x: x if x.geom_type == "LineString" else x.boundary)
        return super().model_post_init(__context)

    def generate_blocks(self, min_block_width=None):

        utils.verbose_print("GENERATING BLOCKS", self.verbose)

        # create a GeoDataFrame with barriers
        barriers = gpd.GeoDataFrame(
            geometry=pd.concat([self.roads.geometry, self.water.geometry, self.railways.geometry]), crs=self.local_crs
        ).reset_index(drop=True)
        barriers = barriers.explode(index_parts=True).reset_index(drop=True).geometry

        # transform enclosed barriers to polygons
        utils.verbose_print("Setting up enclosures...", self.verbose)
        blocks = self._get_enclosures(barriers, self.territory.geometry)

        # fill everything within blocks' boundaries
        utils.verbose_print("Filling holes...", self.verbose)
        blocks = utils.fill_holes(blocks)

        # cleanup after filling holes
        utils.verbose_print("Dropping overlapping blocks...", self.verbose)
        blocks = utils.drop_contained_geometries(blocks)
        blocks = blocks.explode(index_parts=False).reset_index(drop=True)

        blocks = blocks.rename(columns={"index": "block_id"})

        # apply negative and positive buffers consecutively to remove small blocks
        # and divide them on bottlenecks
        if min_block_width is not None:
            utils.verbose_print("Filtering bottlenecks and small blocks...", self.verbose)
            blocks = utils.filter_bottlenecks(blocks, self.local_crs, min_block_width)
            blocks = self._reindex_blocks(blocks)

        # calculate blocks' area in local projected CRS
        utils.verbose_print("Calculating blocks area...", self.verbose)
        blocks["area"] = blocks.to_crs(self.local_crs).area
        blocks = blocks[blocks["area"] > 1]

        # fix blocks' indices
        blocks = self._reindex_blocks(blocks)

        utils.verbose_print("Blocks generated.\n", self.verbose)

        return GeoDataFrame[BlockRow](blocks.to_crs(self.local_crs))

    @staticmethod
    def explore(blocks: GeoDataFrame[BlockRow]):
        return blocks.explore(tiles="CartoDB Positron")

    @staticmethod
    def _get_enclosures(barriers, limit):
        # limit should be a geodataframe or geoseries with with Polygon or MultiPolygon geometry

        barriers = pd.concat([barriers, limit.boundary]).reset_index(drop=True)

        unioned = barriers.unary_union
        polygons = polygonize(unioned)
        enclosures = gpd.GeoSeries(list(polygons), crs=barriers.crs)
        _, enclosure_idxs = enclosures.representative_point().sindex.query(limit.geometry, predicate="contains")
        enclosures = enclosures.iloc[np.unique(enclosure_idxs)]
        enclosures = enclosures.rename("geometry").reset_index()

        return enclosures

    @staticmethod
    def _reindex_blocks(blocks):

        if "block_id" in blocks.columns:
            blocks = blocks.drop("block_id", axis=1).reset_index().rename(columns={"index": "block_id"})
        return blocks
