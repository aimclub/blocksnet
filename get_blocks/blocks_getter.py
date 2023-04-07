"""
Module Desc
"""

from functools import reduce
from typing import Optional

import geopandas as gpd
import osm2geojson
import osmnx as ox
import pandas as pd
import requests
from loguru import logger
from shapely.geometry import Polygon


class BlocksModel:  # pylint: disable=too-few-public-methods,too-many-instance-attributes

    """
    A class used to generate city blocks.
    By default it is initialized for  Peterhof city in Saint Petersburg area with its local crs 32636

    Attributes
    ----------
    city_name : str
        the target city
    city_crs : int
        city's local crs system
    city_admin_level : int
        city's administrative level from OpenStreetMaps

    Methods
    -------
    get_blocks(self)
    """

    GLOBAL_CRS = 4326
    """globally used crs."""
    ROADS_WIDTH = RAILWAYS_WIDTH = NATURE_WIDTH = 3
    """road geometry buffer in meters. So road geometries won't be thin as a line."""
    WATER_WIDTH = 1
    """water geometry buffer in meters. So water geometries in some cases won't be thin as a line."""
    GEOMETRY_CUTOFF_RATIO = 0.15
    """polygon's perimeter to area ratio. Objects with bigger ration will be dropped."""
    GEOMETRY_CUTOFF_AREA = 1_400
    """in meters. Objects with smaller area will be dropped."""
    PARK_CUTOFF_AREA = 10_000
    """in meters. Objects with smaller area will be dropped."""
    OVERPASS_URL = "https://overpass-api.de/api/interpreter"

    def __init__(self, city_name: str = "Петергоф", city_crs: int = 32636, city_admin_level: int = 8):
        self.city_name = city_name
        """city name as a key in a query. City name is the same as city's name in OSM."""
        self.city_crs = city_crs
        """city crs must be specified for more accurate spatial calculations."""
        self.city_admin_level = city_admin_level
        """administrative level must be specified for overpass turbo query."""

        self.water_geometry: Optional[gpd.GeoDataFrame] = None
        self.roads_geometry: Optional[gpd.GeoDataFrame] = None
        self.railways_geometry: Optional[gpd.GeoDataFrame] = None
        self.nature_geometry_boundaries: Optional[gpd.GeoDataFrame] = None
        self.city_geometry: Optional[gpd.GeoDataFrame] = None
        self.city_geometry: Optional[gpd.GeoDataFrame] = None

    def _make_overpass_turbo_request(self, overpass_query, buffer_size: int = 0):
        result = requests.get(
            self.OVERPASS_URL, params={"data": overpass_query}, timeout=600
        ).json()  # pylint: disable=missing-timeout

        # Parse osm response
        resp = osm2geojson.json2geojson(result)
        entity_geometry = (
            gpd.GeoDataFrame.from_features(resp["features"]).set_crs(self.GLOBAL_CRS).to_crs(self.city_crs)
        )
        entity_geometry = entity_geometry[["id", "geometry"]]

        # Output geometry in any case must be some king of Polygon, so it could be extracted from city's geometry
        entity_geometry = entity_geometry.loc[entity_geometry["geometry"].geom_type.isin(["Polygon", "MultiPolygon"])]

        # Buffer geometry in case of line-kind objects like waterways, roads or railways
        if buffer_size:
            entity_geometry["geometry"] = entity_geometry["geometry"].buffer(buffer_size)

        return entity_geometry

    def _get_city_geometry(self) -> None:
        """
        This function downloads the geometry bounds of the chosen city
        and concats it into one solid polygon (or multipolygon)


        Returns
        -------
        city_geometry : GeoDataFrame
            Geometry of city. City geometry is not returned and setted as a class attribute.
        """
        if self.city_geometry:
            logger.info("Got uploaded city geometry")

        else:
            logger.info("City geometry is not provided. Getting water geometry from OSM via overpass turbo")

            overpass_query = f"""
                            [out:json];
                                    area['name'='{self.city_name}']->.searchArea;
                                    (
                                    relation["admin_level"="{self.city_admin_level}"](area.searchArea);
                                    );
                            out geom;
                            """

            self.city_geometry = self._make_overpass_turbo_request(overpass_query=overpass_query)
            self.city_geometry = self.city_geometry.dissolve()

            logger.info("Got city geometry")

    def _fill_spaces_in_blocks(self, row: gpd.GeoSeries) -> Polygon:
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

    def _get_water_geometry(self) -> None:
        """
        This geometry will be cut later from city's geometry.
        The water entities will split blocks from each other. The water geometries are taken using overpass turbo.
        The tags used in the query are: "natural"="water", "waterway"~"river|stream|tidal_channel|canal".


        Returns
        -------
        water_geometry : Union[Polygon, Multipolygon]
            Geometries of water. The water geometries are also buffered a little so the division of city's geometry
            could be more noticable. Water geometry is not returned and setted as a class attribute.
        """

        if self.water_geometry:
            logger.info("Got uploaded water geometries")

        else:
            logger.info("Water geometries are not provided. Getting water geometry from OSM via overpass turbo")

            # Get water polygons in the city
            overpass_query = f"""
                            [out:json];
                                    area['name'='{self.city_name}']->.searchArea;
                                    (
                                    relation["natural"="water"](area.searchArea);
                                    way["natural"="water"](area.searchArea);
                                    relation["waterway"~"river|stream|tidal_channel|canal"](area.searchArea);
                                    way["waterway"~"river|stream|tidal_channel|canal"](area.searchArea);
                                    );
                            out geom;
                            """

            self.water_geometry = self._make_overpass_turbo_request(
                overpass_query=overpass_query, buffer_size=self.WATER_WIDTH
            )

            logger.info("Got water geometries")

    def _get_railways_geometry(self) -> None:
        """
        This geometry will be cut later from city's geometry.
        The railways will split blocks from each other. The railways geometries are taken using overpass turbo.
        The tags used in the query are: "railway"~"rail|light_rail"


        Returns
        -------
        railways_geometry : Union[Polygon, Multipolygon]
            Geometries of railways. Railways geometry is not returned and setted as a class attribute.
        """

        if self.railways_geometry:
            logger.info("Got uploaded railways geometries")

        else:
            logger.info("Railways geometries are not provided. Getting railways geometries from OSM via overpass turbo")

            overpass_query = f"""
                            [out:json];
                                    area['name'='{self.city_name}']->.searchArea;
                                    (
                                    relation["railway"~"rail|light_rail"](area.searchArea);
                                    way["railway"~"rail|light_rail"](area.searchArea);
                                    );
                            out geom;
                            """

            self.railways_geometry = self._make_overpass_turbo_request(
                overpass_query=overpass_query, buffer_size=self.RAILWAYS_WIDTH
            )

            logger.info("Got railways geometries")

    def _get_roads_geometry(self) -> None:
        """
        This geometry will be cut later from city's geometry.
        The roads will split blocks from each other. The road geometries are taken using osmnx.


        Returns
        -------
        self.roads_geometry : Union[Polygon, Multipolygon]
            Geometries of roads buffered by 5 meters so it could be cut from city's geometry
            in the next step. Roads geometry is not returned and setted as a class attribute.
        """

        if self.roads_geometry:
            logger.info("Got uploaded roads geometries")

        else:
            logger.info("Roads geometries are not provided. Getting roads geometries from OSM via OSMNX")

            # Get drive roads from osmnx lib in city bounds
            self.roads_geometry = ox.graph_from_polygon(
                self.city_geometry.to_crs(self.GLOBAL_CRS).geometry.item(), network_type="drive"
            )
            self.roads_geometry = ox.utils_graph.graph_to_gdfs(self.roads_geometry, nodes=False)
            self.roads_geometry = (
                self.roads_geometry.reset_index(level=[0, 1]).reset_index(drop=True).to_crs(self.city_crs)
            )
            self.roads_geometry = self.roads_geometry[["geometry"]]

            # Buffer roads ways to get their close to actual size.
            self.roads_geometry["geometry"] = self.roads_geometry["geometry"].buffer(self.ROADS_WIDTH)

            logger.info("Got roads geometries")

    def _get_nature_geometry(self) -> None:
        """
        This geometry will be cut later from city's geometry.
        Big parks, cemetery and nature_reserve will split blocks from each other. Their geometries are taken using
        overpass turbo.


        Returns
        -------
        self.nature_geometry : Union[Polygon, Multipolygon]
            Geometries of nature entities buffered by 5 meters so it could be cut from city's geometry
            in the next step. Nature geometry is not returned and setted as a class attribute.
        """

        if self.nature_geometry_boundaries:
            logger.info("Got uploaded nature geometries")

        else:
            logger.info("Nature geometries are not provided. Getting nature geometries from OSM via overpass turbo")

            overpass_query_parks = f"""
                            [out:json];
                                    area['name'='{self.city_name}']->.searchArea;
                                    (
                                    relation["leisure"="park"](area.searchArea);
                                    );
                            out geom;
                            """

            overpass_query_greeners = f"""
                            [out:json];
                                    area['name'='{self.city_name}']->.searchArea;
                                    (
                                    relation["landuse"="cemetery"](area.searchArea);
                                    relation["leisure"="nature_reserve"](area.searchArea);
                                    );
                            out geom;
                            """

            self.nature_geometry_boundaries = self._make_overpass_turbo_request(overpass_query=overpass_query_parks)
            self.nature_geometry_boundaries = self.nature_geometry_boundaries[
                self.nature_geometry_boundaries.area > self.PARK_CUTOFF_AREA
            ]
            other_greeners_tmp = self._make_overpass_turbo_request(overpass_query=overpass_query_greeners)
            self.nature_geometry_boundaries = pd.concat([self.nature_geometry_boundaries, other_greeners_tmp])
            del other_greeners_tmp

            self.nature_geometry_boundaries["geometry"] = self.nature_geometry_boundaries.boundary.buffer(
                self.NATURE_WIDTH
            )
            # self.nature_geometry_boundaries.rename(columns={0:"geometry"}, inplace=True)

            logger.info("Got nature geometries")

    def fill_deadends(self) -> None:
        """
        Some roads make a deadend in the block. To get rid of such deadends the blocks' polygons
        are buffered and then unbuffered back. This procedure makes possible to fill deadends.


        Returns
        -------
        self.city_geometry : GeoDataFrame
            Geometry of the city without road deadends. City geometry is not returned and setted as a class attribute.
        """

        logger.info("Starting: filling deadends")
        # To make multi-part geometries into several single-part so they coud be processed separatedly
        self.city_geometry = self.city_geometry.explode(ignore_index=True)
        self.city_geometry["geometry"] = self.city_geometry["geometry"].map(
            lambda block: block.buffer(self.ROADS_WIDTH + 1).buffer(-(self.ROADS_WIDTH + 1))
        )
        logger.info("Finished: filling deadends")

    def cut_railways(self) -> None:
        """
        Subtract railways from city's polygon
        """

        logger.info("Starting: cutting railways geometries")
        self._get_railways_geometry()
        self.city_geometry = gpd.overlay(self.city_geometry, self.railways_geometry, how="difference")
        self.railways_geometry = None
        logger.info("Finished: cutting railways geometries")

    def cut_roads(self) -> None:
        """
        Subtract roads from city's polygon
        """

        logger.info("Starting: cutting roads geometries")
        self._get_roads_geometry()
        self.city_geometry = gpd.overlay(self.city_geometry, self.roads_geometry, how="difference")
        self.roads_geometry = None
        logger.info("Finished: cutting roads geometries")

    def cut_nature(self) -> None:
        """
        Subtract parks, cemetery and other nature from city's polygon.
        This nature entities are substracted only by their boundaries since the could divide polygons into important
        parts for master-planning
        """

        logger.info("Starting: cutting nature geometries")
        self._get_nature_geometry()
        self.city_geometry = gpd.overlay(self.city_geometry, self.nature_geometry_boundaries, how="difference")
        self.nature_geometry_boundaries = None
        logger.info("Finished: cutting nature geometries")

    def cut_water(self) -> None:
        """
        Subtract water from city's polygon.
        Water must be substracted after filling road deadends so small cutted water polygons would be kept
        """

        logger.info("Starting: cutting water geometries")
        self._get_water_geometry()
        self.city_geometry = gpd.overlay(self.city_geometry, self.water_geometry, how="difference")
        self.water_geometry = None
        logger.info("Starting: cutting water geometries")

    def drop_overlayed_geometries(self) -> None:
        """
        Drop overlayed geometries
        """

        logger.info("Starting: dropping overlayed geometries")
        new_geometries = self.city_geometry.unary_union
        new_geometries = gpd.GeoDataFrame(new_geometries, geometry=0)
        self.city_geometry["geometry"] = new_geometries[0]
        del new_geometries
        logger.info("Finished: dropping overlayed geometries")

    def fix_blocks_geometries(self):
        """
        After cutting several entities from city's geometry, blocks might have unnecessary spaces inside them.
        In order to avoid this, this functions prepares data to fill empty spaces inside each city block
        """

        logger.info("Starting: fixing blocks' geometries")
        self.city_geometry = self.city_geometry.explode(ignore_index=True)
        self.city_geometry["rings"] = self.city_geometry.interiors
        self.city_geometry["geometry"] = self.city_geometry[["geometry", "rings"]].apply(
            self._fill_spaces_in_blocks, axis="columns"
        )
        logger.info("Finished: fixing blocks' geometries")

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

        self.cut_railways()
        self.cut_roads()
        self.cut_nature()
        self.fill_deadends()
        self.cut_water()
        self.fix_blocks_geometries()
        self.drop_overlayed_geometries()

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

        logger.info("Dropping unnecessary geometries")

        self.city_geometry["area"] = self.city_geometry["geometry"].area

        # First criteria check: total area
        self.city_geometry = self.city_geometry[self.city_geometry["area"] > self.GEOMETRY_CUTOFF_AREA]

        # Second criteria check: perimetr / total area ratio
        self.city_geometry["length"] = self.city_geometry["geometry"].length
        self.city_geometry["ratio"] = self.city_geometry["length"] / self.city_geometry["area"]

        # Drop polygons with an aspect ratio less than the threshold
        self.city_geometry = self.city_geometry[self.city_geometry["ratio"] < self.GEOMETRY_CUTOFF_RATIO]
        self.city_geometry = self.city_geometry.loc[:, ["id", "geometry"]]

    def get_blocks(self):
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
            a GeoDataFrame of city blocks in setted local crs (32636 by default)
        """

        self._get_city_geometry()
        self._split_city_geometry()
        self._drop_unnecessary_geometries()

        logger.info("Saving output GeoDataFrame with blocks")

        return self.city_geometry
