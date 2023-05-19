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


class ProvisionModel:
    """
    This class represents a model for calculating the provision of a specified service in a city.

    Attributes:
        blocks (gpd.GeoDataFrame): A GeoDataFrame containing information about the blocks in the city.
        service_name (str): The name of the service for which the provision is being calculated.
        city_crs (int): The coordinate reference system used by the city.
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

    def __init__(
        self,
        city_model,
        service_name: str = "schools",
    ):
        self.blocks = city_model.city_blocks
        self.service_name = service_name
        self.city_crs = city_model.city_crs
        self.standard = self.standard_dict[self.service_name]
        self.graph = city_model.services_graphs[self.service_name]

    def get_stats(self):
        """
        This function prints statistics about the blocks in the `g` attribute of the object. The statistics include the number of
        blocks with the service specified by the `service_name` attribute, the number of residential blocks, the total number of blocks,
        and the number of blocks with errors.

        Returns:
            None
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
                elif graph.nodes[key][f"is_{self.service_name}_service"] == 1:
                    blocks_service += 1
            except KeyError:
                invalid_blocks += 1

        print(f"количество кварталов c сервисом {self.service_name}: {blocks_service}")
        print(f"количество жилых кварталов: {blocks_living}")
        print(f"количество кварталов всего: {total}")
        print(f"количество кварталов c ошибкой: {invalid_blocks}")

    def get_provision(self):
        """
        This function calculates the provision of a specified service in a city. The provision is calculated based on the data in the `g` attribute of the object.

        Returns:
            nx.Graph: A networkx graph representing the city's road network with updated data about the provision of the specified service.
        """

        graph = self.graph
        standard = self.standard

        for node in graph.nodes:
            if graph.nodes[node][f"is_{self.service_name}_service"] == 1:
                neighbors = list(graph.neighbors(node))
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
                        capacity -= load
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
                            capacity -= capacity
                            graph.nodes[node][f"id_{self.service_name}"] = node
                            graph.nodes[node][f"population_prov_{self.service_name}"] += prov_people
                            graph.nodes[node][f"population_unprov_{self.service_name}"] = (
                                graph.nodes[node][f"population_unprov_{self.service_name}"] - prov_people
                            )
                            graph.nodes[node][f"id_{self.service_name}"] = node
                            graph.nodes[node][f"provision_{self.service_name}"] = (prov_people * 100) / graph.nodes[node][
                                "population"
                            ]

                for neighbor in neighbors:
                    if (
                        graph.nodes[neighbor]["is_living"]
                        and graph.nodes[neighbor]["population"] > 0
                        and graph.nodes[neighbor][f"is_{self.service_name}_service"] == 0
                        and capacity > 0
                    ):
                        if (
                            graph.nodes[neighbor]["is_living"]
                            and graph.nodes[neighbor]["population"] > 0
                            and graph.nodes[neighbor][f"provision_{self.service_name}"] < 100
                        ):
                            if graph.nodes[neighbor][f"provision_{self.service_name}"] == 0:
                                load = (graph.nodes[neighbor][f"population_unprov_{self.service_name}"] / 1000) * standard

                            elif graph.nodes[neighbor][f"provision_{self.service_name}"] > 0:
                                load = (graph.nodes[neighbor][f"population_unprov_{self.service_name}"] / 1000) * standard

                            if load <= capacity:
                                capacity -= load
                                graph.nodes[neighbor][f"provision_{self.service_name}"] = 100
                                graph.nodes[neighbor][f"id_{self.service_name}"] = node
                                graph.nodes[neighbor][f"population_prov_{self.service_name}"] += graph.nodes[neighbor][
                                    f"population_unprov_{self.service_name}"
                                ]
                                graph.nodes[neighbor][f"population_unprov_{self.service_name}"] -= graph.nodes[neighbor][
                                    f"population_unprov_{self.service_name}"
                                ]

                            else:
                                if capacity > 0:
                                    prov_people = (capacity * 1000) / standard
                                    capacity -= capacity

                                    graph.nodes[neighbor][f"id_{self.service_name}"] = neighbor
                                    graph.nodes[neighbor][f"population_prov_{self.service_name}"] += prov_people
                                    graph.nodes[neighbor][f"population_unprov_{self.service_name}"] = (
                                        graph.nodes[neighbor][f"population_unprov_{self.service_name}"] - prov_people
                                    )
                                    graph.nodes[neighbor][f"id_{self.service_name}"] = node
                                    graph.nodes[neighbor][f"provision_{self.service_name}"] = (prov_people * 100) / graph.nodes[
                                        neighbor
                                    ]["population"]

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
        # blocks.index = blocks.index.astype(str)
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
                    blocks.loc[indx, f"provision_{self.service_name}"] = graph.nodes[node][f"provision_{self.service_name}"]
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
