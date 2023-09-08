"""
This module is aimed to cut the geometry of the specified city into city blocks.
Blocks are close to cadastral unit, however in the way they are calculated in this method
they become suitable to be used in masterplanning process.

TODO: add landuse devision to avoid weird cutoffs
"""

import geopandas as gpd
from enum import Enum
from pydantic import BaseModel
from typing import Literal

from masterplan_tools.models.geojson import PolygonGeoJSON
from .cut_parameters import CutParameters
from .land_use_parameters import LandUseParameters
from .landuse_filter import LuFilter
from .blocks_clustering import BlocksClusterization
from .utils import Utils


class BlocksCutterFeatureProperties(BaseModel):
    id: int
    landuse: Literal["no_dev_area", "selected_area", "buildings"] = "selected_area"
    # development: bool = True


class BlocksCutter(BaseModel):  # pylint: disable=too-few-public-methods,too-many-instance-attributes

    """
    A class used to generate city blocks.

    Methods
    -------
    get_blocks(self)
    """

    cut_parameters: CutParameters
    lu_parameters: LandUseParameters | None = None

    def _fill_deadends(self, blocks: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Some roads make a deadend in the blocks. To get rid of such deadends the blocks' polygons
        are buffered and then unbuffered back. This procedure makes possible to fill deadends.


        Returns
        -------
        self.cut_parameters.city : GeoDataFrame
            Geometry of the city without road deadends. City geometry is not returned and setted as a class attribute.
        """

        # To make multi-part geometries into several single-part so they coud be processed separatedly
        blocks = blocks.explode(ignore_index=True)
        blocks["geometry"] = blocks["geometry"].map(
            lambda block: block.buffer(self.cut_parameters.roads_buffer + 1).buffer(
                -(self.cut_parameters.roads_buffer + 1)
            )
        )
        return blocks

    def cut_blocks_by_polygons(self, blocks: gpd.GeoDataFrame, *polygons: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Cut any geometries from blocks' geometries

        Returns
        -------
        None
        """
        result = gpd.GeoDataFrame(data=blocks)
        for polygon in polygons:
            polygon = Utils.polygon_to_multipolygon(polygon)
            result = gpd.overlay(result, polygon, how="difference")
        return result

    def _drop_overlayed_geometries(self, blocks) -> None:
        """
        Drop overlayed geometries

        Returns
        -------
        None
        """

        new_geometries = blocks.unary_union
        new_geometries = gpd.GeoDataFrame(geometry=[new_geometries], crs=blocks.crs.to_epsg())
        blocks["geometry"] = new_geometries.loc[:, "geometry"]
        return blocks

    def _cut_blocks(self) -> gpd.GeoDataFrame:
        """
        Gets city geometry to split it by dividers like different kind of roads. The splitted parts are city blocks.
        However, not each resulted geometry is a valid city block. So inaccuracies of this division would be removed
        in the next step.

        Returns
        -------
        blocks : Union[Polygon, Multipolygon]
            city bounds splitted by railways, roads and water. Resulted polygons are city blocks
        """
        blocks = self.cut_blocks_by_polygons(
            self.cut_parameters.city.to_gdf(), self.cut_parameters.railways.to_gdf(), self.cut_parameters.roads.to_gdf()
        )
        blocks = self._fill_deadends(blocks)
        blocks = self.cut_blocks_by_polygons(blocks, self.cut_parameters.water.to_gdf())
        blocks = Utils._fix_blocks_geometries(blocks)
        blocks = self._drop_overlayed_geometries(blocks)
        blocks = blocks.explode(index_parts=True).reset_index()[["geometry"]]
        return blocks

    def get_blocks(self) -> PolygonGeoJSON[BlocksCutterFeatureProperties]:
        """
        Main method.

        This method gets city's boundaries from OSM. Then iteratively water, roads and railways are cutted
        from city's geometry. After splitting city into blocks invalid blocks with bad geometries are removed.

        For big city it takes about ~ 1-2 hours to split big city's geometry into thousands of blocks.
        It takes so long because splitting one big geometry (like city) by another big geometry (like roads
        or waterways) is computationally expensive. Splitting city by roads and water entities are two the most
        time consuming processes in this method.

        For cities over the Russia it could be better to upload water geometries from elsewhere, than taking
        them from OSM. Somehow they are poorly documented on OSM.


        Returns
        -------
        blocks
            a GeoDataFrame of city blocks
        """

        blocks = self._cut_blocks()
        if self.lu_parameters != None:
            blocks = LuFilter(blocks, landuse_geometries=self.lu_parameters).filter_lu()
            if self.lu_parameters.buildings != None:
                blocks = BlocksClusterization(blocks, self.lu_parameters).run()
        blocks.reset_index(inplace=True)
        blocks["id"] = blocks.index
        if "landuse" in blocks:
            blocks["development"] = blocks["landuse"] != "no_dev_area"
        new_geometries = blocks.unary_union
        new_geometries = gpd.GeoDataFrame(geometry=[new_geometries], crs=blocks.crs.to_epsg())
        new_blocks = new_geometries.explode(index_parts=True).reset_index()[['geometry']]
        blocks = gpd.sjoin(new_blocks, blocks, how='inner', predicate='intersects').drop_duplicates('geometry')
        blocks = blocks.drop(['index_right', 'index', 'id'], axis=1)
        blocks = blocks.reset_index(names='id')
        return PolygonGeoJSON[BlocksCutterFeatureProperties].from_gdf(blocks)
