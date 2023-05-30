"""
This module is aimed to cut the geometry of the specified city into city blocks.
Blocks are close to cadastral unit, however in the way they are calculated in this method
they become suitable to be used in masterplanning process.

TODO: add landuse devision to avoid weird cutoffs
"""

from functools import reduce
from typing import Optional

import geopandas as gpd
from shapely.geometry import MultiPolygon, Polygon


class BlocksCutter:  # pylint: disable=too-few-public-methods,too-many-instance-attributes

    """
    A class used to generate city blocks.

    Methods
    -------
    get_blocks(self)
    """

    def __init__(self, city_model):
        self.roads_buffer = city_model.ROADS_WIDTH
        """roads geometry buffer in meters"""
        self.geometry_cutoff_ration = city_model.GEOMETRY_CUTOFF_RATIO
        """polygon's perimeter to area ratio. Objects with bigger ration will be dropped."""
        self.geometry_cutoff_area = city_model.GEOMETRY_CUTOFF_AREA
        """in meters. Objects with smaller area will be dropped."""
        self.park_cutoff_area = city_model.PARK_CUTOFF_AREA
        """in meters. Objects with smaller area will be dropped."""

        self.water_geometry: Optional[gpd.GeoDataFrame] = city_model.water_geometry
        self.roads_geometry: Optional[gpd.GeoDataFrame] = city_model.roads_geometry
        self.railways_geometry: Optional[gpd.GeoDataFrame] = city_model.railways_geometry
        self.nature_geometry_boundaries: Optional[gpd.GeoDataFrame] = city_model.nature_geometry_boundaries
        self.city_geometry: Optional[gpd.GeoDataFrame] = city_model.city_geometry

    @staticmethod
    def _fill_spaces_in_blocks(row: gpd.GeoSeries) -> Polygon:
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

        if len(row["rings"]) > 0:
            empty_part_to_fill = [Polygon(ring) for ring in row["rings"]]

            if len(empty_part_to_fill) > 0:
                new_block_geometry = reduce(
                    lambda geom1, geom2: geom1.union(geom2), [row["geometry"]] + empty_part_to_fill
                )

        if new_block_geometry:
            return new_block_geometry

        return row["geometry"]

    def _fill_deadends(self) -> None:
        """
        Some roads make a deadend in the blocks. To get rid of such deadends the blocks' polygons
        are buffered and then unbuffered back. This procedure makes possible to fill deadends.


        Returns
        -------
        self.city_geometry : GeoDataFrame
            Geometry of the city without road deadends. City geometry is not returned and setted as a class attribute.
        """

        # To make multi-part geometries into several single-part so they coud be processed separatedly
        self.city_geometry = self.city_geometry.explode(ignore_index=True)
        self.city_geometry["geometry"] = self.city_geometry["geometry"].map(
            lambda block: block.buffer(self.roads_buffer + 1).buffer(-(self.roads_buffer + 1))
        )

    @staticmethod
    def _polygon_to_multipolygon(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        This function makes one multipolygon from many polygons in the gdf.

        This step allows to fasten overlay operation between two multipolygons since overlay operates
        ineratively by passed geometry objects.

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

    def _cut_city_by_polygons(self, gdf_cutter) -> None:
        """
        Cut any geometries from city's geometry

        Returns
        -------
        None
        """

        gdf_cutter = self._polygon_to_multipolygon(gdf_cutter)
        self.city_geometry = self._polygon_to_multipolygon(self.city_geometry)

        self.city_geometry = gpd.overlay(self.city_geometry, gdf_cutter, how="difference")

    def _drop_overlayed_geometries(self) -> None:
        """
        Drop overlayed geometries

        Returns
        -------
        None
        """

        new_geometries = self.city_geometry.unary_union
        new_geometries = gpd.GeoDataFrame(geometry=[new_geometries])
        self.city_geometry["geometry"] = new_geometries.loc[:, "geometry"]
        del new_geometries

    def _fix_blocks_geometries(self) -> None:
        """
        After cutting several entities from city's geometry, blocks might have unnecessary spaces inside them.

        In order to avoid this, this functions prepares data to fill empty spaces inside each city block

        Returns
        -------
        None
        """

        self.city_geometry = self.city_geometry.explode(ignore_index=True)
        self.city_geometry["rings"] = self.city_geometry.interiors
        self.city_geometry["geometry"] = self.city_geometry[["geometry", "rings"]].apply(
            self._fill_spaces_in_blocks, axis="columns"
        )

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

        self._cut_city_by_polygons(self.railways_geometry)
        self._cut_city_by_polygons(self.roads_geometry)
        self._cut_city_by_polygons(self.nature_geometry_boundaries)
        self._fill_deadends()
        self._cut_city_by_polygons(self.water_geometry)
        self._fix_blocks_geometries()
        self._drop_overlayed_geometries()

        self.city_geometry = self.city_geometry.reset_index()[["index", "geometry"]].rename(columns={"index": "id"})
        self.city_geometry = self.city_geometry.explode(ignore_index=True)

    def _drop_unnecessary_geometries(self) -> None:
        """
        Get and clear blocks's geometries, dropping unnecessary geometries.

        There two criterieas to dicide whether drop the geometry or not.

        First -- if the total area of the polygon is not too small (not less than self.cutoff_area)

        Second -- if the ratio of perimeter to total polygon area not more than self.cutoff_ratio.

        The second criteria means that polygon is not too narrow and could be a valid block.

        A GeoDataFrame with blocks and unnecessary geometries such as roundabouts,
        other small polygons or very narrow geometries which happened to exist after
        cutting roads and water from the city's polygon, but have no value for master-planning
        purposes


        Returns
        -------
        blocks : GeoDataFrame
            a GeoDataFrame with substracted blocks without unnecessary geometries.
            Blocks geometry is not returned and setted as a class attribute.
        """
        self.city_geometry["area"] = self.city_geometry["geometry"].area

        # First criteria check: total area
        self.city_geometry = self.city_geometry[self.city_geometry["area"] > self.geometry_cutoff_area]

        # Second criteria check: perimetr / total area ratio
        self.city_geometry["length"] = self.city_geometry["geometry"].length
        self.city_geometry["ratio"] = self.city_geometry["length"] / self.city_geometry["area"]

        # Drop polygons with an aspect ratio less than the threshold
        self.city_geometry = self.city_geometry[self.city_geometry["ratio"] < self.geometry_cutoff_ration]
        self.city_geometry = self.city_geometry.loc[:, ["id", "geometry"]]

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

        self._split_city_geometry()
        self._drop_unnecessary_geometries()

        self.city_geometry.drop(columns=["id"], inplace=True)
        self.city_geometry = self.city_geometry.reset_index(drop=True).reset_index(drop=False)
        self.city_geometry.rename(columns={"index": "id"}, inplace=True)

        return self.city_geometry
