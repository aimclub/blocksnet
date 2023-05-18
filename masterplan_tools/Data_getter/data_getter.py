"""
This module gets data from the OSM. The module also implements several methods of data processing.
These methods allow you to connect parts of the data processing pipeline.
"""


import geopandas as gpd  # pylint: disable=import-error
import osm2geojson  # pylint: disable=import-error
import osmnx as ox  # pylint: disable=import-error
import pandas as pd
import requests
from loguru import logger  # pylint: disable=import-error
import networkx as nx
from tqdm.auto import tqdm  # pylint: disable=import-error

from masterplan_tools.Data_getter.accs_matrix_calculator import Accessibility


class DataGetter:
    """
    This class is used to get and pre-process data to be used in calculations in other modules.
    """

    GLOBAL_CRS = 4326
    """this crs is used in osm by default"""
    OVERPASS_URL = "https://overpass-api.de/api/interpreter"
    """stable working api link for overpass turbo"""
    HECTARE = 10000
    """hectares in meters"""

    def __init__(self, city_crs: int = 32636) -> None:
        self.city_crs = city_crs
        """city's crs system; must be specified; by default crs is set for Saint Petersburg, Russia"""

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

    def get_buildings(self, engine=None, city_crs=None, city_id=None, from_device=True):
        """
        This function returns a GeoDataFrame containing information about buildings in a city. The data can be read
        from a local file or from a PostGIS database.

        Args:
            engine (sqlalchemy.engine.Engine, optional): A SQLAlchemy engine object used to connect to the PostGIS database. Defaults to None.
            city_crs (int, optional): The coordinate reference system used by the city. Defaults to None.
            city_id (int, optional): The ID of the city in the PostGIS database. Defaults to None.
            from_device (bool, optional): If True, the data is read from a local file. If False, the data is read from the PostGIS database.
            Defaults to True.

        Returns:
            gpd.GeoDataFrame: A GeoDataFrame containing information about buildings in the city.
        """

        if from_device:
            df_buildings = gpd.read_parquet("../masterplanning/masterplan_tools/output_data/buildings.parquet")

        else:
            df_buildings = gpd.read_postgis(
                f"select population_balanced, building_area, living_area, storeys_count, is_living, "
                f"ST_Centroid(ST_Transform(geometry, {city_crs})) as geom from all_buildings where city_id={city_id}",
                con=engine,
            )
            df_buildings.rename(columns={"geom": "geometry"}, inplace=True)
        return df_buildings

    def get_service(self, service_type=None, city_crs=None, engine=None, city_id=None, from_device=False):
        """
        This function returns a GeoDataFrame containing information about blocks with a specified service in a city.
        The data can be read from a local file or from a PostGIS database.

        Args:
            service_type (str, optional): The type of service to retrieve data for. Defaults to None.
            city_crs (int, optional): The coordinate reference system used by the city. Defaults to None.
            engine (sqlalchemy.engine.Engine, optional): A SQLAlchemy engine object used to connect to the PostGIS database. Defaults to None.
            city_id (int, optional): The ID of the city in the PostGIS database. Defaults to None.
            from_device (bool, optional): If True, the data is read from a local file. If False, the data is read from the PostGIS database.
            Defaults to False.

        Returns:
            gpd.GeoDataFrame: A GeoDataFrame containing information about blocks with the specified service in the city.
        """
        if from_device:
            service_blocks_df = gpd.read_parquet(
                f"../masterplanning/masterplan_tools/output_data/{service_type}.parquet"
            )

        else:
            service_blocks_df = gpd.read_postgis(
                f"select capacity, geometry as geom from all_services where city_service_type_code in ('{service_type}') "
                f"and city_id={city_id}",
                con=engine,
            )

            service_blocks_df.rename(columns={"geom": "geometry"}, inplace=True)
            service_blocks_df["geometry"] = service_blocks_df["geometry"].convex_hull
            service_blocks_df["geometry"] = service_blocks_df["geometry"].to_crs(city_crs)
            service_blocks_df["geometry"] = service_blocks_df["geometry"].centroid
            service_blocks_df["geometry"] = service_blocks_df["geometry"].set_crs(city_crs)
            service_blocks_df = service_blocks_df.set_geometry("geometry")

        return service_blocks_df

    def get_accessibility_matrix(self, city_crs=None, blocks=None, G=None, option=None):
        """
        This function returns an accessibility matrix for a city. The matrix is calculated using the `Accessibility` class.

        Args:
            city_crs (int, optional): The coordinate reference system used by the city. Defaults to None.
            blocks (gpd.GeoDataFrame, optional): A GeoDataFrame containing information about the blocks in the city. Defaults to None.
            G (nx.Graph, optional): A networkx graph representing the city's road network. Defaults to None.
            option (str, optional): An option specifying how the accessibility matrix should be calculated. Defaults to None.

        Returns:
            np.ndarray: An accessibility matrix for the city.
        """

        accessibility = Accessibility(city_crs, blocks, G, option)
        return accessibility.get_matrix()

    def get_greenings(self, engine, city_id, city_crs):
        """
        This function returns a GeoDataFrame containing information about green spaces in a city. The data is read from a PostGIS database.

        Args:
            engine (sqlalchemy.engine.Engine): A SQLAlchemy engine object used to connect to the PostGIS database.
            city_id (int): The ID of the city in the PostGIS database.
            city_crs (int): The coordinate reference system used by the city.

        Returns:
            gpd.GeoDataFrame: A GeoDataFrame containing information about green spaces in the city.
        """

        greenings = gpd.read_postgis(
            f"select ROUND(ST_Area(geometry::geography))::int as current_green_area, "
            f"capacity as current_green_capacity, ST_Centroid(ST_Transform(geometry, {city_crs})) as geom "
            f"from all_services "
            f"where city_service_type_code like 'recr%' and city_id={city_id}",
            con=engine,
        )

        return greenings

    def get_parkings(self, engine, city_id, city_crs):
        """
        This function returns a GeoDataFrame containing information about parking spaces in a city. The data is read from a PostGIS database.

        Args:
            engine (sqlalchemy.engine.Engine): A SQLAlchemy engine object used to connect to the PostGIS database.
            city_id (int): The ID of the city in the PostGIS database.
            city_crs (int): The coordinate reference system used by the city.

        Returns:
            gpd.GeoDataFrame: A GeoDataFrame containing information about parking spaces in the city.
        """

        parkings = gpd.read_postgis(
            f"select capacity as current_parking_capacity, "
            f"ST_Centroid(ST_Transform(geometry, {city_crs})) as geom "
            f"from all_services "
            f"where service_name like 'Парковка' "
            f"and city_id={city_id}",
            con=engine,
        )

        return parkings

    def _get_living_area(self, row):
        """
        This function calculates the living area of a building based on the data in the given row.

        Args:
            row (pd.Series): A row of data containing information about a building.

        Returns:
            float: The calculated living area of the building.
        """

        if row["living_area"]:
            return row["living_area"]
        else:
            if row["is_living"]:
                if row["storeys_count"]:
                    if row["building_area"]:
                        living_area = row["building_area"] * row["storeys_count"] * 0.7

                        return living_area
                    else:
                        return 0
                else:
                    return 0
            else:
                return 0

    def _get_living_area_pyatno(self, row):
        """
        This function calculates the living area of a building based on the data in the given row. If the `living_area` attribute is
        not available, the function returns 0.

        Args:
            row (pd.Series): A row of data containing information about a building.

        Returns:
            float: The calculated living area of the building.
        """

        if row["living_area"]:
            return row["building_area"]
        else:
            return 0

    def aggregate_blocks_info(self, blocks, buildings, greenings, parkings):
        """
        This function aggregates information about blocks in a city. The information includes data about buildings, green spaces,
        and parking spaces.

        Args:
            blocks (gpd.GeoDataFrame): A GeoDataFrame containing information about the blocks in the city.
            buildings (gpd.GeoDataFrame): A GeoDataFrame containing information about buildings in the city.
            greenings (gpd.GeoDataFrame): A GeoDataFrame containing information about green spaces in the city.
            parkings (gpd.GeoDataFrame): A GeoDataFrame containing information about parking spaces in the city.

        Returns:
            gpd.GeoDataFrame: A GeoDataFrame containing aggregated information about blocks in the city.
        """

        buildings["living_area"].fillna(0, inplace=True)
        buildings["storeys_count"].fillna(0, inplace=True)
        buildings["living_area"] = buildings.progress_apply(self._get_living_area, axis=1)
        buildings["living_area_pyatno"] = buildings.progress_apply(self._get_living_area_pyatno, axis=1)
        buildings["total_area"] = buildings["building_area"] * buildings["storeys_count"]

        blocks_and_greens = (
            gpd.sjoin(blocks, greenings, predicate="intersects", how="left")
            .groupby("id")
            .agg(
                {
                    "current_green_capacity": "sum",
                    "current_green_area": "sum",
                }
            )
        )
        blocks_and_greens = (
            blocks_and_greens.reset_index(drop=True).reset_index(drop=False).rename(columns={"index": "block_id"})
        )

        blocks_and_parkings = (
            gpd.sjoin(blocks, parkings, predicate="intersects", how="left")
            .groupby("id")
            .agg({"current_parking_capacity": "sum"})
        )
        blocks_and_parkings = (
            blocks_and_parkings.reset_index(drop=True).reset_index(drop=False).rename(columns={"index": "block_id"})
        )

        blocks_and_buildings = (
            gpd.sjoin(blocks, buildings, predicate="intersects", how="left")
            .drop(columns=["index_right"])
            .groupby("id")
            .agg(
                {
                    "population_balanced": "sum",
                    "building_area": "sum",
                    "storeys_count": "median",
                    "total_area": "sum",
                    "living_area": "sum",
                    "living_area_pyatno": "sum",
                }
            )
        )
        blocks_and_buildings = (
            blocks_and_buildings.reset_index(drop=True).reset_index(drop=False).rename(columns={"index": "block_id"})
        )

        blocks.reset_index(drop=False, inplace=True)

        blocks_info_aggregated = pd.merge(blocks_and_buildings, blocks_and_greens)
        blocks_info_aggregated = pd.merge(blocks_info_aggregated, blocks_and_parkings)

        blocks_info_aggregated = gpd.GeoDataFrame(
            pd.merge(blocks, blocks_info_aggregated, left_on="index", right_on="block_id").drop(
                columns=["index", "id"]
            ),
            geometry="geometry",
        )
        blocks_info_aggregated.rename(
            columns={"building_area": "building_area_pyatno", "total_area": "building_area"}, inplace=True
        )

        blocks_info_aggregated["current_industrial_area"] = (
            blocks_info_aggregated["building_area_pyatno"] - blocks_info_aggregated["living_area_pyatno"]
        )
        blocks_info_aggregated.rename(
            columns={
                "population_balanced": "current_population",
                "storeys_count": "floors",
                "living_area_pyatno": "current_living_area",
            },
            inplace=True,
        )
        blocks_info_aggregated["area"] = blocks_info_aggregated["geometry"].area
        blocks_info_aggregated.drop(columns=["building_area_pyatno", "building_area", "living_area"], inplace=True)

        return blocks_info_aggregated

    def prepare_graph(
        self,
        blocks,
        city_crs,
        service_type=None,
        service_gdf=None,
        accessibility_matrix=None,
        buildings=None,
        updated_block_info=None,
    ):
        """
        This function prepares a graph for calculating the provision of a specified service in a city.

        Args:
            blocks (gpd.GeoDataFrame): A GeoDataFrame containing information about the blocks in the city.
            city_crs (int): The coordinate reference system used by the city.
            service_type (str, optional): The type of service to calculate the provision for. Defaults to None.
            service_gdf (gpd.GeoDataFrame, optional): A GeoDataFrame containing information about blocks with the specified service in the city.
            Defaults to None.
            accessibility_matrix (np.ndarray, optional): An accessibility matrix for the city. Defaults to None.
            buildings (gpd.GeoDataFrame, optional): A GeoDataFrame containing information about buildings in the city. Defaults to None.
            updated_block_info (gpd.GeoDataFrame, optional): A GeoDataFrame containing updated information about blocks in the city.
            Defaults to None.

        Returns:
            nx.Graph: A networkx graph representing the city's road network with additional data for calculating the provision of the
            specified service.
        """
        services_accessibility = {
            "kindergartens": 4,
            "schools": 7,
            "universities": 60,
            "hospitals": 60,
            "policlinics": 13,
            "theaters": 60,
            "cinemas": 60,
            "cafes": 30,
            "bakeries": 30,
            "fastfoods": 30,
            "music_school": 30,
            "sportgrounds": 7,
            "swimming_pools": 30,
            "conveniences": 8,
            "recreational_areas": 25,
            "pharmacies": 7,
            "playgrounds": 2,
            "supermarkets": 30,
        }

        accs_time = services_accessibility[service_type]
        service = service_gdf

        blocks_with_buildings = (
            gpd.sjoin(blocks, buildings, predicate="intersects", how="left")
            .drop(columns=["index_right"])
            .groupby("id")
            .agg({"population_balanced": "sum", "living_area": "sum"})
        )

        blocks_with_buildings.reset_index(drop=False, inplace=True)
        blocks_with_buildings["is_living"] = blocks_with_buildings["living_area"].apply(
            lambda x: True if x > 0 else False
        )
        # blocks_with_buildings.rename(columns={"living_area": "is_living"}, inplace=True)

        blocks = blocks_with_buildings.merge(blocks, right_on="id", left_on="id")
        blocks = gpd.GeoDataFrame(blocks, geometry="geometry", crs=city_crs)

        living_blocks = blocks.loc[:, ["id", "geometry"]].sort_values(by="id").reset_index(drop=True)

        service_gdf = (
            gpd.sjoin(blocks, service, predicate="intersects")
            .groupby("id")
            .agg(
                {
                    "capacity": "sum",
                }
            )
        )
        # print(service_blocks_df)
        if updated_block_info:
            print(service_gdf.loc[updated_block_info["block_id"], "capacity"])
            service_gdf.loc[updated_block_info["block_id"], "capacity"] += updated_block_info[
                f"{service_type}_capacity"
            ]
            print(service_gdf.loc[updated_block_info["block_id"], "capacity"])

            blocks.loc[updated_block_info["block_id"], "population_balanced"] = updated_block_info["population"]

        blocks_geom_dict = blocks[["id", "population_balanced", "is_living"]].set_index("id").to_dict()
        service_blocks_dict = service_gdf.to_dict()["capacity"]

        blocks_list = accessibility_matrix.loc[
            accessibility_matrix.index.isin(service_gdf.index.astype("Int64")),
            accessibility_matrix.columns.isin(living_blocks["id"]),
        ]

        service_graph = nx.Graph()

        for idx in tqdm(list(blocks_list.index)):
            blocks_list_tmp = blocks_list[blocks_list.index == idx]
            blocks_list.columns = blocks_list.columns.astype(int)
            blocks_list_tmp = blocks_list_tmp[blocks_list_tmp < accs_time].dropna(axis=1)
            blocks_list_tmp_dict = blocks_list_tmp.transpose().to_dict()[idx]

            for key in blocks_list_tmp_dict.keys():
                if key != idx:
                    service_graph.add_edge(idx, key, weight=round(blocks_list_tmp_dict[key], 1))

                else:
                    service_graph.add_node(idx)

                service_graph.nodes[key]["population"] = blocks_geom_dict["population_balanced"][int(key)]
                service_graph.nodes[key]["is_living"] = blocks_geom_dict["is_living"][int(key)]

                if key != idx:
                    try:
                        if service_graph.nodes[key][f"is_{service_type}_service"] != 1:
                            service_graph.nodes[key][f"is_{service_type}_service"] = 0
                            service_graph.nodes[key][f"provision_{service_type}"] = 0
                            service_graph.nodes[key][f"id_{service_type}"] = 0
                    except KeyError:
                        service_graph.nodes[key][f"is_{service_type}_service"] = 0
                        service_graph.nodes[key][f"provision_{service_type}"] = 0
                        service_graph.nodes[key][f"id_{service_type}"] = 0
                else:
                    service_graph.nodes[key][f"is_{service_type}_service"] = 1
                    service_graph.nodes[key][f"{service_type}_capacity"] = service_blocks_dict[key]
                    service_graph.nodes[key][f"provision_{service_type}"] = 0
                    service_graph.nodes[key][f"id_{service_type}"] = 0

                if service_graph.nodes[key]["is_living"]:
                    service_graph.nodes[key][f"population_prov_{service_type}"] = 0
                    service_graph.nodes[key][f"population_unprov_{service_type}"] = blocks_geom_dict[
                        "population_balanced"
                    ][int(key)]

        return service_graph

    def balance_data(self, gdf, polygon, school, kindergarten, greening):
        """
        This function balances data about blocks in a city by intersecting the given GeoDataFrame with a polygon and calculating various statistics.

        Args:
            gdf (gpd.GeoDataFrame): A GeoDataFrame containing information about blocks in the city.
            polygon (gpd.GeoSeries): A polygon representing the area to intersect with the blocks.
            school (gpd.GeoDataFrame): A GeoDataFrame containing information about schools in the city.
            kindergarten (gpd.GeoDataFrame): A GeoDataFrame containing information about kindergartens in the city.
            greening (gpd.GeoDataFrame): A GeoDataFrame containing information about green spaces in the city.

        Returns:
            dict: A dictionary containing balanced data about blocks in the city.
        """

        intersecting_blocks = gpd.overlay(gdf, polygon, how="intersection").drop(columns=["id"])
        intersecting_blocks.rename(columns={"id_1": "id"}, inplace=True)
        gdf = intersecting_blocks

        gdf["current_building_area"] = gdf["current_living_area"] + gdf["current_industrial_area"]
        gdf_ = gdf[
            [
                "block_id",
                "area",
                "current_living_area",
                "current_industrial_area",
                "current_population",
                "current_green_area",
                "floors",
            ]
        ]

        gdf_ = (
            gdf_.merge(school[["id", "population_unprov_schools"]], left_on="block_id", right_on="id")
            .merge(kindergarten[["id", "population_unprov_kindergartens"]], left_on="block_id", right_on="id")
            .merge(greening[["id", "population_unprov_recreational_areas"]], left_on="block_id", right_on="id")
        )
        gdf_.drop(["id_x", "id_y", "id"], axis=1, inplace=True)

        gdf_["area"] = gdf_["area"] / self.HECTARE
        gdf_["current_living_area"] = gdf_["current_living_area"] / self.HECTARE
        gdf_["current_industrial_area"] = gdf_["current_industrial_area"] / self.HECTARE
        gdf_["current_green_area"] = gdf_["current_green_area"] / self.HECTARE

        df_sum = gdf_.sum()
        df_sum["floors"] = gdf_["floors"].mean()
        df_new = pd.DataFrame(df_sum).T

        sample = df_new[df_new["area"] > 7].sample()
        sample = sample.to_dict("records")
        block = sample[0].copy()

        return block
