"""
Module Desc
"""


import geopandas as gpd  # pylint: disable=import-error
import osm2geojson  # pylint: disable=import-error
import osmnx as ox  # pylint: disable=import-error
import pandas as pd
import requests
from loguru import logger  # pylint: disable=import-error
from shapely.geometry import Polygon  # pylint: disable=import-error
import networkx as nx
from tqdm.auto import tqdm # pylint: disable=import-error

from masterplan_tools.Data_getter.accs_matrix_calculator import Accessibility


class DataGetter:
    """
    TODO: add docstring
    """

    GLOBAL_CRS = 4326
    OVERPASS_URL = "https://overpass-api.de/api/interpreter"

    def __init__(self, city_crs: int = 32636) -> None:
        self.city_crs = city_crs

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

        # Buffer roads ways to get their close to actual size.
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
        # self.nature_geometry_boundaries.rename(columns={0:"geometry"}, inplace=True)

        logger.info("Got nature geometries")

        return nature_geometry_boundaries

    def get_buildings(self, engine=None, city_crs=None, city_id=None, from_device=True):
        """
        TODO: add docstring
        """

        if from_device:
            df_buildings = gpd.read_parquet("/home/gk/jupyter/masterplanning/mp_tools/output_data/buildings.parquet")

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
        TODO: add docstring
        """

        if from_device:
            service_blocks_df = gpd.read_parquet(
                f"/home/gk/jupyter/masterplanning/mp_tools/output_data/{service_type}.parquet"
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
        TODO: add docstring
        """
        
        accessibility = Accessibility(city_crs, blocks, G, option)
        return accessibility.get_matrix()

    def get_greenings(self, engine, city_id, city_crs):
        """
        TODO: add docstring
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
        TODO: add docstring
        """

        parkings = gpd.read_postgis(
            f"select capacity as current_parking_capacity, "
            f"ST_Centroid(ST_Transform(geometry, {city_crs})) as geom "
            f"from all_services "
            f"where service_type like 'Парковка' "
            f"and city_id={city_id}",
            con=engine,
        )

        return parkings

    def _get_living_area(self, row):
        """
        TODO: add docstring
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
        TODO: add docstring
        """

        if row["living_area"]:
            return row["building_area"]
        else:
            return 0

    def aggregate_blocks_info(self, blocks, engine, city_id, city_crs, from_device=False):
        """
        TODO: add docstring
        """

        buildings = self.get_buildings(engine=engine, city_id=city_id, city_crs=city_crs, from_device=from_device)
        greenings = self.get_greenings(engine=engine, city_id=city_id, city_crs=city_crs)
        parkings = self.get_parkings(engine=engine, city_id=city_id, city_crs=city_crs)

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
        blocks_and_greens = blocks_and_greens.reset_index(drop=True).reset_index(drop=False).rename(columns={"index": "block_id"})

        blocks_and_parkings = (
            gpd.sjoin(blocks, parkings, predicate="intersects", how="left")
            .groupby("id")
            .agg({"current_parking_capacity": "sum"})
        )
        blocks_and_parkings = blocks_and_parkings.reset_index(drop=True).reset_index(drop=False).rename(columns={"index": "block_id"})

        blocks_and_buildings = (
            gpd.sjoin(blocks, buildings, predicate="intersects", how="left")
            .drop(columns=['index_right'])
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
        blocks_and_buildings = blocks_and_buildings.reset_index(drop=True).reset_index(drop=False).rename(columns={"index": "block_id"})

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


    def prepare_graph(self, blocks, engine, city_id, city_crs, from_device=False, service_type=None, updated_block_info = None, 
                      accessibility_matrix = None):
        """
        TODO: add docstring
        """
        services_accessibility = {"kindergartens":4,"schools":7,"universities":60,"hospitals":60,
        "policlinics":13,"theaters":60,"cinemas":60,"cafes":30,"bakeries":30,"fastfoods":30, "music_school":30, "sportgrounds":7,
        "swimming_pools":30,"conveniences":8,"recreational_areas":25,"pharmacies":7,"playgrounds":2,"supermarkets":30}

        accs_time = services_accessibility[service_type]

        buildings = self.get_buildings(engine=engine, city_id=city_id, city_crs=city_crs, from_device=from_device)
        service = self.get_service(self, service_type=service_type, city_crs=city_crs, engine=engine, city_id=city_id, from_device=from_device)

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
            gpd.sjoin(blocks,  service, predicate="intersects")
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
                f'{service_type}_capacity'
            ]
            print(service_gdf.loc[updated_block_info["block_id"], "capacity"])

            blocks.loc[updated_block_info["block_id"], "population_balanced"] = updated_block_info[
                "population"
            ]

        blocks_geom_dict = blocks[["id", "population_balanced", "is_living"]].set_index("id").to_dict()
        service_blocks_dict = service_gdf.to_dict()["capacity"]

        blocks_list = accessibility_matrix.loc[
            accessibility_matrix.index.isin(service_gdf.index.astype("Int64")),
            accessibility_matrix.columns.isin(living_blocks["id"]),
        ]

        g = nx.Graph()

        for idx in tqdm(list(blocks_list.index)):
            blocks_list_tmp = blocks_list[blocks_list.index == idx]
            blocks_list.columns = blocks_list.columns.astype(int)
            blocks_list_tmp = blocks_list_tmp[blocks_list_tmp < accs_time].dropna(axis=1)
            blocks_list_tmp_dict = blocks_list_tmp.transpose().to_dict()[idx]

            for key in blocks_list_tmp_dict.keys():

                if key != idx:
                    g.add_edge(idx, key, weight=round(blocks_list_tmp_dict[key], 1))

                else:
                    
                    g.add_node(idx)

                g.nodes[key]['population'] = blocks_geom_dict['population_balanced'][int(key)]
                g.nodes[key]['is_living'] = blocks_geom_dict['is_living'][int(key)]

                if  key != idx:
                    try:
                        if g.nodes[key][f'is_{service_type}_service'] != 1:
                            g.nodes[key][f'is_{service_type}_service'] = 0
                            g.nodes[key][f'provision_{service_type}'] = 0
                            g.nodes[key][f'id_{service_type}'] = 0
                    except KeyError:
                        g.nodes[key][f'is_{service_type}_service'] = 0
                        g.nodes[key][f'provision_{service_type}'] = 0
                        g.nodes[key][f'id_{service_type}'] = 0
                else:
                    g.nodes[key][f'is_{service_type}_service'] = 1
                    g.nodes[key][f'{service_type}_capacity'] = service_blocks_dict[key]
                    g.nodes[key][f'provision_{service_type}'] = 0
                    g.nodes[key][f'id_{service_type}'] = 0

                if g.nodes[key]['is_living'] == True:
                    g.nodes[key][f'population_prov_{service_type}'] = 0
                    g.nodes[key][f'population_unprov_{service_type}'] = blocks_geom_dict['population_balanced'][int(key)]

        return g