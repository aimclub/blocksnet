"""
This module is aimed to provide all necessary tools to get an estimation of provision of the selected service
"""


import pandas as pd
import numpy as np
import psycopg2 as pg
import geopandas as gpd
import networkx as nx
from tqdm.auto import tqdm  # pylint: disable=import-error

tqdm.pandas()

from masterplan_tools.Data_getter.data_getter import DataGetter


class ProvisionModel:
    """
    This class represents a model for calculating the provision of a specified service in a city.

    Attributes:
        blocks (gpd.GeoDataFrame): A GeoDataFrame containing information about the blocks in the city.
        service_name (str): The name of the service for which the provision is being calculated.
        standard (int): The standard value for the specified service, taken from the `standard_dict` attribute.
        g (int): The value of the `g` attribute.
    """

    standard_dict = {
        "kindergartens": 61,
        "schools": 120,
        "universities": 13,
        "hospitals": 9,
        "policlinics": 27,
        "theaters": 5,
        "cinemas": 10,
        "cafes": 72,
        "bakeries": 72,
        "fastfoods": 72,
        "music_school": 8,
        "sportgrounds": 15,
        "swimming_pools": 50,
        "conveniences": 90,
        "recreational_areas": 5000,
        "pharmacies": 50,
        "playgrounds": 550,
        "supermarkets": 992,
    }

    services_accessibility_dict = {
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

    def __init__(
        self,
        city_model,
        service_name: str = "schools",
    ):
        self.blocks = city_model.city_blocks.copy()
        self.service_name = service_name
        self.standard = self.standard_dict[self.service_name]
        self.accessibility = self.services_accessibility_dict[self.service_name]
        self.graph = city_model.services_graph.copy()
        self.blocks_aggregated = city_model.blocks_aggregated_info.copy()

    def get_stats(self):
        """
        This function prints statistics about the blocks in the `g` attribute of the object. The statistics include the number of
        blocks with the service specified by the `service_name` attribute, the number of residential blocks, the total number of blocks,
        and the number of blocks with errors.

        Returns:
            stats
        """

        graph = self.graph.copy()
        if not graph:
            return 0

        blocks_service = 0
        invalid_blocks = 0
        blocks_living = 0
        total = 0
        for key in graph:
            total += 1
            try:
                if graph.nodes[key]["is_living"]:
                    blocks_living += 1
                if graph.nodes[key][f"is_{self.service_name}_service"] == 1:
                    blocks_service += 1
            except KeyError:
                invalid_blocks += 1

        print(f"Number of blocks with service: {self.service_name}: {blocks_service}")
        print(f"Number of residential blocks: {blocks_living}")
        print(f"Number of blocks total: {total}")
        print(f"Number of blocks with an error: {invalid_blocks}")

    def get_provision(self):
        """
        This function calculates the provision of a specified service in a city. The provision is calculated based on the data in the `g` attribute of the object.

        Returns:
            nx.Graph: A networkx graph representing the city's road network with updated data about the provision of the specified service.
        """

        graph = self.graph.copy()
        standard = self.standard
        accessibility = self.accessibility

        for u, v, data in list(graph.edges(data=True)):
            if data["weight"] > accessibility:
                graph.remove_edge(u, v)

        for node in list(graph.nodes):
            if graph.degree(node) == 0 and graph.nodes[node][f"is_{self.service_name}_service"] != 1:
                graph.remove_node(node)

        for node in graph.nodes:
            if graph.nodes[node][f"is_{self.service_name}_service"] == 1:
                capacity = graph.nodes[node][f"{self.service_name}_capacity"]
                if (
                    graph.nodes[node]["is_living"]
                    and graph.nodes[node]["population"] > 0
                    and graph.nodes[node][f"provision_{self.service_name}"] < 100
                ):
                    if graph.nodes[node][f"provision_{self.service_name}"] == 0:
                        load = (graph.nodes[node][f"population_unprov_{self.service_name}"] / 1000) * standard

                    elif graph.nodes[node][f"provision_{self.service_name}"] > 0:
                        load = (graph.nodes[node][f"population_unprov_{self.service_name}"] / 1000) * standard

                    if load <= capacity:
                        graph.nodes[node][f"{self.service_name}_capacity"] -= load
                        graph.nodes[node][f"provision_{self.service_name}"] = 100
                        graph.nodes[node][f"id_{self.service_name}"] = node
                        graph.nodes[node][f"population_prov_{self.service_name}"] += graph.nodes[node][
                            f"population_unprov_{self.service_name}"
                        ]
                        graph.nodes[node][f"population_unprov_{self.service_name}"] -= graph.nodes[node][
                            f"population_unprov_{self.service_name}"
                        ]

                    else:
                        if capacity > 0:
                            prov_people = (capacity * 1000) / standard
                            graph.nodes[node][f"{self.service_name}_capacity"] -= capacity
                            graph.nodes[node][f"id_{self.service_name}"] = node
                            graph.nodes[node][f"population_prov_{self.service_name}"] += prov_people
                            graph.nodes[node][f"population_unprov_{self.service_name}"] = (
                                graph.nodes[node][f"population_unprov_{self.service_name}"] - prov_people
                            )
                            graph.nodes[node][f"id_{self.service_name}"] = node
                            graph.nodes[node][f"provision_{self.service_name}"] = (prov_people * 100) / graph.nodes[
                                node
                            ]["population"]

        for node in graph.nodes:
            if graph.nodes[node][f"is_{self.service_name}_service"] == 1:
                capacity = graph.nodes[node][f"{self.service_name}_capacity"]
                neighbors = list(graph.neighbors(node))
                for neighbor in neighbors:
                    if graph.nodes[neighbor]["is_living"] and graph.nodes[neighbor]["population"] > 0 and capacity > 0:
                        if (
                            graph.nodes[neighbor]["is_living"]
                            and graph.nodes[neighbor]["population"] > 0
                            and graph.nodes[neighbor][f"provision_{self.service_name}"] < 100
                        ):
                            if graph.nodes[neighbor][f"provision_{self.service_name}"] == 0:
                                load = (
                                    graph.nodes[neighbor][f"population_unprov_{self.service_name}"] / 1000
                                ) * standard

                            elif graph.nodes[neighbor][f"provision_{self.service_name}"] > 0:
                                load = (
                                    graph.nodes[neighbor][f"population_unprov_{self.service_name}"] / 1000
                                ) * standard

                            if load <= capacity:
                                graph.nodes[node][f"{self.service_name}_capacity"] -= load
                                graph.nodes[neighbor][f"provision_{self.service_name}"] = 100
                                graph.nodes[neighbor][f"id_{self.service_name}"] = node
                                graph.nodes[neighbor][f"population_prov_{self.service_name}"] += graph.nodes[neighbor][
                                    f"population_unprov_{self.service_name}"
                                ]
                                graph.nodes[neighbor][f"population_unprov_{self.service_name}"] -= graph.nodes[
                                    neighbor
                                ][f"population_unprov_{self.service_name}"]

                            else:
                                if capacity > 0:
                                    prov_people = (capacity * 1000) / standard
                                    graph.nodes[node][f"{self.service_name}_capacity"] -= capacity

                                    graph.nodes[neighbor][f"id_{self.service_name}"] = neighbor
                                    graph.nodes[neighbor][f"population_prov_{self.service_name}"] += prov_people
                                    graph.nodes[neighbor][f"population_unprov_{self.service_name}"] = (
                                        graph.nodes[neighbor][f"population_unprov_{self.service_name}"] - prov_people
                                    )
                                    graph.nodes[neighbor][f"id_{self.service_name}"] = node
                                    graph.nodes[neighbor][f"provision_{self.service_name}"] = (
                                        prov_people * 100
                                    ) / graph.nodes[neighbor]["population"]

        self.graph = graph

    def set_blocks_attributes(self):
        """
        This function returns a copy of the `blocks` attribute of the object with updated values for the service specified by
        the `service_name` attribute. The values are updated based on the data in the `g` attribute of the object.

        Returns:
            DataFrame: A copy of the `blocks` attribute with updated values for the specified service.
        """
        graph = self.graph.copy()
        blocks = self.blocks.copy()
        blocks_aggregated = self.blocks_aggregated
        blocks[f"provision_{self.service_name}"] = 0
        blocks[f"id_{self.service_name}"] = 0
        blocks[f"population_prov_{self.service_name}"] = 0
        blocks[f"population_unprov_{self.service_name}"] = 0
        blocks[f"provision_{self.service_name}"] = 0
        blocks["population"] = 0

        for node in graph:
            indx = blocks[blocks.index == node].index[0]
            if graph.nodes[node]["is_living"]:
                if graph.nodes[node].get(f"id_{self.service_name}") is not None:
                    blocks.loc[indx, f"id_{self.service_name}"] = graph.nodes[node][f"id_{self.service_name}"]
                    blocks.loc[indx, f"population_prov_{self.service_name}"] = graph.nodes[node][
                        f"population_prov_{self.service_name}"
                    ]
                    blocks.loc[indx, f"population_unprov_{self.service_name}"] = graph.nodes[node][
                        f"population_unprov_{self.service_name}"
                    ]
                    blocks.loc[indx, f"provision_{self.service_name}"] = graph.nodes[node][
                        f"provision_{self.service_name}"
                    ]
                    blocks.loc[indx, "population"] = graph.nodes[node]["population"]

                else:
                    blocks[f"population_unprov_{self.service_name}"][indx] = graph.nodes[node][
                        f"population_unprov_{self.service_name}"
                    ]

        blocks[f"id_{self.service_name}"] = blocks[f"id_{self.service_name}"].astype(int)
        blocks[f"population_prov_{self.service_name}"] = blocks[f"population_prov_{self.service_name}"].astype(int)
        blocks[f"population_unprov_{self.service_name}"] = blocks[f"population_unprov_{self.service_name}"].astype(int)
        blocks[f"provision_{self.service_name}"] = blocks[f"provision_{self.service_name}"].astype(int)
        blocks["population"] = blocks["population"].astype(int)

        for i in range(len(blocks)):
            if blocks.loc[i, "population"] == 0:
                blocks.loc[i, "population"] = blocks_aggregated.loc[i, "current_population"]
                blocks.loc[i, f"population_unprov_{self.service_name}"] = blocks_aggregated.loc[i, "current_population"]

        blocks = blocks.drop(columns=["index"])
        blocks = blocks.drop(columns=[f"id_{self.service_name}"])

        return blocks

    def run(self):
        """
        This function runs the model to calculate the provision of a specified service in a city. The function calls the `get_stats`, `get_provision`, and `get_geo` methods of the object.

        Returns:
            DataFrame: A DataFrame containing information about the provision of the specified service in the city.
        """

        self.get_stats()
        self.get_provision()
        return self.set_blocks_attributes()
