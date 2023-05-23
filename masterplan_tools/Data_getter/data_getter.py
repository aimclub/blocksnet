"""
This module gets data from the OSM. The module also implements several methods of data processing.
These methods allow you to connect parts of the data processing pipeline.
"""


import geopandas as gpd  # pylint: disable=import-error
import pandas as pd
from tqdm.auto import tqdm  # pylint: disable=import-error

from masterplan_tools.Data_getter.accs_matrix_calculator import Accessibility


class DataGetter:
    """
    This class is used to get and pre-process data to be used in calculations in other modules.
    """

    HECTARE = 10000
    """hectares in meters"""

    def __init__(self) -> None:
        pass

    def get_accessibility_matrix(self, blocks=None, G=None):
        """
        This function returns an accessibility matrix for a city. The matrix is calculated using the `Accessibility` class.

        Args:
            blocks (gpd.GeoDataFrame, optional): A GeoDataFrame containing information about the blocks in the city. Defaults to None.
            G (nx.Graph, optional): A networkx graph representing the city's road network. Defaults to None.

        Returns:
            np.ndarray: An accessibility matrix for the city.
        """

        accessibility = Accessibility(blocks, G)
        return accessibility.get_matrix()

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
        service_type=None,
        service_gdf=None,
        accessibility_matrix=None,
        buildings=None,
        updated_block_info=None,
        services_graph=None,
    ):
        """
        This function prepares a graph for calculating the provision of a specified service in a city.

        Args:
            blocks (gpd.GeoDataFrame): A GeoDataFrame containing information about the blocks in the city.
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
        service = service_gdf

        blocks_with_buildings = (
            gpd.sjoin(blocks, buildings, predicate="intersects", how="left")
            .drop(columns=["index_right"])
            .groupby("id")
            .agg({"population_balanced": "sum"})
        )

        blocks_with_buildings.reset_index(drop=False, inplace=True)
        blocks_with_buildings["is_living"] = blocks_with_buildings["population_balanced"].apply(
            lambda x: True if x > 0 else False
        )

        blocks_crs = blocks.crs.to_epsg()

        blocks = blocks_with_buildings.merge(blocks, right_on="id", left_on="id")
        blocks = gpd.GeoDataFrame(blocks, geometry="geometry", crs=blocks_crs)

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

        if updated_block_info:
            if service_type == "recreational_areas":
                service_gdf.loc[updated_block_info["block_id"], "capacity"] += updated_block_info[f"G_max_capacity"]
            else:
                service_gdf.loc[updated_block_info["block_id"], "capacity"] += updated_block_info[
                    f"{service_type}_capacity"
                ]
            blocks.loc[updated_block_info["block_id"], "population_balanced"] = updated_block_info["population"]

        blocks_geom_dict = blocks[["id", "population_balanced", "is_living"]].set_index("id").to_dict()
        service_blocks_dict = service_gdf.to_dict()["capacity"]

        blocks_list = accessibility_matrix.loc[
            accessibility_matrix.index.isin(service_gdf.index.astype("Int64")),
            accessibility_matrix.columns.isin(living_blocks["id"]),
        ]

        for idx in tqdm(list(blocks_list.index)):
            blocks_list_tmp = blocks_list[blocks_list.index == idx]
            blocks_list.columns = blocks_list.columns.astype(int)
            # blocks_list_tmp = blocks_list_tmp[blocks_list_tmp < accs_time].dropna(axis=1)
            blocks_list_tmp_dict = blocks_list_tmp.transpose().to_dict()[idx]

            for key in blocks_list_tmp_dict.keys():
                if key != idx:
                    services_graph.add_edge(idx, key, weight=round(blocks_list_tmp_dict[key], 1))

                else:
                    services_graph.add_node(idx)

                services_graph.nodes[key]["population"] = blocks_geom_dict["population_balanced"][int(key)]
                services_graph.nodes[key]["is_living"] = blocks_geom_dict["is_living"][int(key)]

                if key != idx:
                    try:
                        if services_graph.nodes[key][f"is_{service_type}_service"] != 1:
                            services_graph.nodes[key][f"is_{service_type}_service"] = 0
                            services_graph.nodes[key][f"provision_{service_type}"] = 0
                            services_graph.nodes[key][f"id_{service_type}"] = 0
                    except KeyError:
                        services_graph.nodes[key][f"is_{service_type}_service"] = 0
                        services_graph.nodes[key][f"provision_{service_type}"] = 0
                        services_graph.nodes[key][f"id_{service_type}"] = 0
                else:
                    services_graph.nodes[key][f"is_{service_type}_service"] = 1
                    services_graph.nodes[key][f"{service_type}_capacity"] = service_blocks_dict[key]
                    services_graph.nodes[key][f"provision_{service_type}"] = 0
                    services_graph.nodes[key][f"id_{service_type}"] = 0

                if services_graph.nodes[key]["is_living"]:
                    services_graph.nodes[key][f"population_prov_{service_type}"] = 0
                    services_graph.nodes[key][f"population_unprov_{service_type}"] = blocks_geom_dict[
                        "population_balanced"
                    ][int(key)]

        return services_graph

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
