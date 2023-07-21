"""
This module gets data from the OSM. The module also implements several methods of data processing.
These methods allow you to connect parts of the data processing pipeline.

TODO: rewrite using some wrapper e.g. overpy or osmnx
"""


import geopandas as gpd
import osm2geojson
import osmnx as ox
import pandas as pd
import requests
from loguru import logger


class DataGetter:
    """
    This class is used to get and pre-process data to be used in calculations in other modules.
    """

    GLOBAL_CRS = 4326
    """this crs is used in osm by default"""
    OVERPASS_URL = "https://overpass-api.de/api/interpreter"
    """stable working api link for overpass turbo"""

    def __init__(self, city_crs) -> None:
        self.city_crs = city_crs

    def _make_overpass_turbo_request(self, overpass_query, buffer_size: int = 0):
        """
        This function makes a request to the Overpass API using the given query and returns a GeoDataFrame containing the resulting data.

        Args:
            overpass_query (str): The Overpass query to use in the request.
            buffer_size (int, optional): The size of the buffer to apply to line-like geometries. Defaults to 0.

        Returns:
            gpd.GeoDataFrame: A GeoDataFrame containing the data returned by the Overpass API.
        """

        result = requests.get(
            self.OVERPASS_URL, params={"data": overpass_query}, timeout=600
        ).json()  # pylint: disable=missing-timeout

        # Parse osm response
        resp = osm2geojson.json2geojson(result)
        # print(gpd.GeoDataFrame.from_features(resp["features"]))
        entity_geometry = (
            gpd.GeoDataFrame.from_features(resp["features"]).set_crs(self.GLOBAL_CRS).to_crs(self.city_crs)
        )
        entity_geometry = entity_geometry[["id", "geometry"]]

        # Output geometry in any case must be some king of Polygon, so it could be extracted from city's geometry
        entity_geometry = entity_geometry.loc[
            entity_geometry["geometry"].geom_type.isin(["Polygon", "MultiPolygon", "LineString", "MultiLineString"])
        ]

        # Buffer geometry in case of line-kind objects like waterways, roads or railways
        if buffer_size:
            entity_geometry["geometry"] = entity_geometry["geometry"].buffer(buffer_size)

        return entity_geometry

    def get_city_geometry(self, city_name, city_admin_level) -> None:
        """
        This function downloads the geometry bounds of the chosen city
        and concats it into one solid polygon (or multipolygon)


        Returns
        -------
        city_geometry : GeoDataFrame
            Geometry of city. City geometry is not returned and setted as a class attribute.
        """

        logger.info("City geometry is not provided. Getting water geometry from OSM via overpass turbo")

        overpass_query = f"""
                        [out:json];
                                area['name'='{city_name}']->.searchArea;
                                (
                                relation["admin_level"="{city_admin_level}"](area.searchArea);
                                );
                        out geom;
                        """

        city_geometry = self._make_overpass_turbo_request(overpass_query=overpass_query)
        city_geometry = city_geometry.dissolve()

        logger.info("Got city geometry")

        return city_geometry

    def get_water_geometry(self, city_name, water_buffer) -> None:
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

        logger.info("Water geometries are not provided. Getting water geometry from OSM via overpass turbo")

        # Get water polygons in the city
        overpass_query = f"""
                        [out:json];
                                area['name'='{city_name}']->.searchArea;
                                (
                                relation["natural"="water"](area.searchArea);
                                way["natural"="water"](area.searchArea);
                                relation["waterway"~"river|stream|tidal_channel|canal"](area.searchArea);
                                way["waterway"~"river|stream|tidal_channel|canal"](area.searchArea);
                                );
                        out geom;
                        """

        water_geometry = self._make_overpass_turbo_request(overpass_query=overpass_query, buffer_size=water_buffer)

        logger.info("Got water geometries")

        return water_geometry

    def get_railways_geometry(self, city_name, railways_buffer) -> None:
        """
        This geometry will be cut later from city's geometry.
        The railways will split blocks from each other. The railways geometries are taken using overpass turbo.
        The tags used in the query are: "railway"~"rail|light_rail"


        Returns
        -------
        railways_geometry : Union[Polygon, Multipolygon]
            Geometries of railways. Railways geometry is not returned and setted as a class attribute.
        """

        logger.info("Railways geometries are not provided. Getting railways geometries from OSM via overpass turbo")

        overpass_query = f"""
                        [out:json];
                                area['name'='{city_name}']->.searchArea;
                                (
                                relation["railway"~"rail|light_rail"](area.searchArea);
                                way["railway"~"rail|light_rail"](area.searchArea);
                                );
                        out geom;
                        """

        railways_geometry = self._make_overpass_turbo_request(
            overpass_query=overpass_query, buffer_size=railways_buffer
        )

        logger.info("Got railways geometries")

        return railways_geometry

    def get_roads_geometry(self, city_geometry, roads_buffer) -> None:
        """
        This geometry will be cut later from city's geometry.
        The roads will split blocks from each other. The road geometries are taken using osmnx.


        Returns
        -------
        self.roads_geometry : Union[Polygon, Multipolygon]
            Geometries of roads buffered by 5 meters so it could be cut from city's geometry
            in the next step. Roads geometry is not returned and setted as a class attribute.
        """

        logger.info("Roads geometries are not provided. Getting roads geometries from OSM via OSMNX")

        # Get drive roads from osmnx lib in city bounds
        roads_geometry = ox.graph_from_polygon(
            city_geometry.to_crs(self.GLOBAL_CRS).geometry.item(), network_type="drive"
        )
        roads_geometry = ox.utils_graph.graph_to_gdfs(roads_geometry, nodes=False)

        roads_geometry = roads_geometry[["geometry"]]

        roads_geometry = roads_geometry.reset_index(level=[0, 1]).reset_index(drop=True).to_crs(self.city_crs)

        roads_geometry["geometry"] = roads_geometry["geometry"].buffer(roads_buffer)

        logger.info("Got roads geometries")

        return roads_geometry

    def get_nature_geometry(self, city_name, nature_buffer, park_cutoff_area) -> None:
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

        logger.info("Nature geometries are not provided. Getting nature geometries from OSM via overpass turbo")

        overpass_query_parks = f"""
                        [out:json];
                                area['name'='{city_name}']->.searchArea;
                                (
                                relation["leisure"="park"](area.searchArea);
                                );
                        out geom;
                        """

        overpass_query_greeners = f"""
                        [out:json];
                                area['name'='{city_name}']->.searchArea;
                                (
                                relation["landuse"="cemetery"](area.searchArea);
                                relation["leisure"="nature_reserve"](area.searchArea);
                                );
                        out geom;
                        """

        nature_geometry_boundaries = self._make_overpass_turbo_request(overpass_query=overpass_query_parks)
        nature_geometry_boundaries = nature_geometry_boundaries[nature_geometry_boundaries.area > park_cutoff_area]
        other_greeners_tmp = self._make_overpass_turbo_request(overpass_query=overpass_query_greeners)
        nature_geometry_boundaries = pd.concat([nature_geometry_boundaries, other_greeners_tmp])
        del other_greeners_tmp

        nature_geometry_boundaries["geometry"] = nature_geometry_boundaries.boundary.buffer(nature_buffer)

        logger.info("Got nature geometries")

        return nature_geometry_boundaries
