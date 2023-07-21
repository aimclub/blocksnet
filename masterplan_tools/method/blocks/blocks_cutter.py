"""
This module is aimed to cut the geometry of the specified city into city blocks.
Blocks are close to cadastral unit, however in the way they are calculated in this method
they become suitable to be used in masterplanning process.

TODO: add landuse devision to avoid weird cutoffs
"""

from functools import reduce

import geopandas as gpd
from pydantic import BaseModel
from shapely.geometry import MultiPolygon, Polygon

from masterplan_tools.models.geojson import GeoJSON
from .blocks_cutter_geometries import BlocksCutterGeometries, BlocksCutterFeature
from .blocks_cutter_parameters import BlocksCutterParameters


class BlocksCutter(BaseModel):  # pylint: disable=too-few-public-methods,too-many-instance-attributes

    """
    A class used to generate city blocks.

    Methods
    -------
    get_blocks(self)
    """

    parameters: BlocksCutterParameters = BlocksCutterParameters()
    geometries: BlocksCutterGeometries

    @staticmethod
    def _fill_spaces_in_blocks(block: gpd.GeoSeries) -> Polygon:
        """
        This geometry will be cut later from city's geometry.
        The water entities will split blocks from each other. The water geometries are taken using overpass turbo.
        The tags used in the query are: "natural"="water", "waterway"~"river|stream|tidal_channel|canal".
        This function is designed to be used with 'apply' function, applied to the pd.DataFrame.


        Parameters
        ----------
        row : GeoSeries


        Returns
        -------
        water_geometry : Union[Polygon, Multipolygon]
            Geometry of water. The water geometries are also buffered a little so the division of city's geometry
            could be more noticable.
        """

        new_block_geometry = None

        if len(block["rings"]) > 0:
            empty_part_to_fill = [Polygon(ring) for ring in block["rings"]]

            if len(empty_part_to_fill) > 0:
                new_block_geometry = reduce(
                    lambda geom1, geom2: geom1.union(geom2), [block["geometry"]] + empty_part_to_fill
                )

        if new_block_geometry:
            return new_block_geometry

        return block["geometry"]

    def _fill_deadends(self, blocks: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Some roads make a deadend in the blocks. To get rid of such deadends the blocks' polygons
        are buffered and then unbuffered back. This procedure makes possible to fill deadends.


        Returns
        -------
        self.geometries.city : GeoDataFrame
            Geometry of the city without road deadends. City geometry is not returned and setted as a class attribute.
        """

        # To make multi-part geometries into several single-part so they coud be processed separatedly
        blocks = blocks.explode(ignore_index=True)
        blocks["geometry"] = blocks["geometry"].map(
            lambda block: block.buffer(self.parameters.roads_buffer + 1).buffer(-(self.parameters.roads_buffer + 1))
        )
        return blocks

    @staticmethod
    def _polygon_to_multipolygon(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        This function makes one multipolygon from many polygons in the gdf.
        This step allows to fasten overlay operation between two multipolygons since overlay operates ineratively
        by passed geometry objects.

        Attributes
        ----------
        gdf: gpd.GeoDataFrame
            A gdf with many geometries-polygons

        Returns
        -------
        gdf: Multipolygon
            A gdf with one geometry-MultiPolygon
        """

        crs = gdf.crs
        gdf = gdf.unary_union
        if isinstance(gdf, Polygon):
            gdf = gpd.GeoDataFrame(geometry=[gdf], crs=crs)
        else:
            gdf = gpd.GeoDataFrame(geometry=[MultiPolygon(gdf)], crs=crs)
        return gdf

    def _cut_blocks_by_polygons(self, blocks: gpd.GeoDataFrame, polygons: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Cut any geometries from blocks' geometries

        Returns
        -------
        None
        """

        polygons = self._polygon_to_multipolygon(polygons)
        return gpd.overlay(blocks, polygons, how="difference")

        self.city_geometry = gpd.overlay(self.city_geometry, gdf_cutter, how="difference", keep_geom_type=True)

    def _drop_overlayed_geometries(self) -> None:
        """
        Drop overlayed geometries

        Returns
        -------
        None
        """

        new_geometries = self.city_geometry.unary_union
        new_geometries = gpd.GeoDataFrame(geometry=[new_geometries], crs=self.city_geometry.crs.to_epsg())
        self.city_geometry["geometry"] = new_geometries.loc[:, "geometry"]

    @staticmethod
    def _fix_blocks_geometries(city_geometry):
        """
        After cutting several entities from city's geometry, blocks might have unnecessary spaces inside them.
        In order to avoid this, this functions prepares data to fill empty spaces inside each city block

        Returns
        -------
        None
        """

        city_geometry = city_geometry.explode(ignore_index=True)
        city_geometry["rings"] = city_geometry.interiors
        city_geometry["geometry"] = city_geometry[["geometry", "rings"]].apply(
            BlocksCutter._fill_spaces_in_blocks, axis="columns"
        )

        return city_geometry

    def _split_city_geometry(self) -> gpd.GeoDataFrame:
        """
        Gets city geometry to split it by dividers like different kind of roads. The splitted parts are city blocks.
        However, not each resulted geometry is a valid city block. So inaccuracies of this division would be removed
        in the next step.

        Returns
        -------
        blocks : Union[Polygon, Multipolygon]
            city bounds splitted by railways, roads and water. Resulted polygons are city blocks
        """
        blocks = self._cut_blocks_by_polygons(self.geometries.city.to_gdf(), self.geometries.railways.to_gdf())
        blocks = self._cut_blocks_by_polygons(blocks, self.geometries.roads.to_gdf())
        blocks = self._cut_blocks_by_polygons(blocks, self.geometries.nature.to_gdf())
        blocks = self._fill_deadends(blocks)
        blocks = self._cut_blocks_by_polygons(blocks, self.geometries.water.to_gdf())
        blocks = self._fix_blocks_geometries(blocks)
        blocks = self._drop_overlayed_geometries(blocks)

        self._cut_city_by_polygons(self.railways_geometry)
        self._cut_city_by_polygons(self.roads_geometry)
        self._fill_deadends()
        self._cut_city_by_polygons(self.water_geometry)
        self._cut_city_by_polygons(self.no_dev_zone)
        self._cut_city_by_polygons(self.landuse_zone)
        self.city_geometry = self._fix_blocks_geometries(self.city_geometry)
        self._drop_overlayed_geometries()

    def get_blocks(self) -> gpd.GeoDataFrame:
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

        blocks = self._split_city_geometry()
        blocks = blocks.explode(index_parts=True).reset_index()[["geometry"]]

        return GeoJSON[BlocksCutterFeature].from_gdf(blocks)
