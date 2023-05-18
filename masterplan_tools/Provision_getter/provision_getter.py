import pandas as pd
import numpy as np
import psycopg2 as pg
import geopandas as gpd
import networkx as nx
from tqdm.auto import tqdm  # pylint: disable=import-error

tqdm.pandas()


class ProvisionModel:
    """
    TODO: add docstring
    TODO: manage UserWarning: pandas only supports SQLAlchemy connectable
    """

    def __init__(
        self,
        blocks: gpd.GeoDataFrame,
        matrix: pd.DataFrame,
        buildings: gpd.GeoDataFrame,
        service_gdf: gpd.GeoDataFrame,
        SUB_GROUP: str = "schools",
        ACCS_TIME=30,
        standard=61,
        city_crs: int = 32636,
        updated_block_info: dict = None,
    ):
        self.updated_block_info = updated_block_info
        self.blocks = blocks
        self.matrix = matrix
        self.buildings = buildings
        self.SUB_GROUP = SUB_GROUP
        self.ACCS_TIME = ACCS_TIME
        self.standard = standard
        self.city_crs = city_crs
        self.service_gdf = service_gdf

    def prepare_graph(self):
        """
        TODO: add docstring
        """
        blocks = self.blocks.copy()

        blocks_with_buildings = (
            gpd.sjoin(blocks, self.buildings, predicate="intersects", how="left")
            .drop(columns=["index_right"])
            .groupby("id")
            .agg({"population_balanced": "sum", "living_area": "sum"})
        )

        blocks_with_buildings.reset_index(drop=False, inplace=True)
        blocks_with_buildings["living_area"] = blocks_with_buildings["living_area"].apply(
            lambda x: True if x > 0 else False
        )
        blocks_with_buildings.rename(columns={"living_area": "is_living"}, inplace=True)

        blocks = blocks_with_buildings.merge(blocks, right_on="id", left_on="id")
        blocks = gpd.GeoDataFrame(blocks, geometry="geometry", crs=self.city_crs)

        living_blocks = blocks.loc[:, ["id", "geometry"]].sort_values(by="id").reset_index(drop=True)

        self.service_gdf = (
            gpd.sjoin(blocks, self.service_gdf, predicate="intersects")
            .groupby("id")
            .agg(
                {
                    "capacity": "sum",
                }
            )
        )
        # print(service_blocks_df)
        if self.updated_block_info:
            print(self.service_gdf.loc[self.updated_block_info["block_id"], "capacity"])
            self.service_gdf.loc[self.updated_block_info["block_id"], "capacity"] += self.updated_block_info[
                f"{self.SUB_GROUP}_capacity"
            ]
            print(self.service_gdf.loc[self.updated_block_info["block_id"], "capacity"])

            blocks.loc[self.updated_block_info["block_id"], "population_balanced"] = self.updated_block_info[
                "population"
            ]

        blocks_geom_dict = blocks[["id", "population_balanced", "is_living"]].set_index("id").to_dict()
        service_blocks_dict = self.service_gdf.to_dict()["capacity"]

        blocks_list = self.matrix.loc[
            self.matrix.index.isin(self.service_gdf.index.astype("Int64")),
            self.matrix.columns.isin(living_blocks["id"]),
        ]

        self.g = nx.Graph()

        for idx in tqdm(list(blocks_list.index)):
            blocks_list_tmp = blocks_list[blocks_list.index == idx]
            blocks_list.columns = blocks_list.columns.astype(int)
            blocks_list_tmp = blocks_list_tmp[blocks_list_tmp < self.ACCS_TIME].dropna(axis=1)
            blocks_list_tmp_dict = blocks_list_tmp.transpose().to_dict()[idx]

            for key in blocks_list_tmp_dict.keys():

                if key != idx:
                    self.g.add_edge(idx, key, weight=round(blocks_list_tmp_dict[key], 1))

                else:

                    self.g.add_node(idx)

                self.g.nodes[key]["population"] = blocks_geom_dict["population_balanced"][int(key)]
                self.g.nodes[key]["is_living"] = blocks_geom_dict["is_living"][int(key)]

                if key != idx:
                    try:
                        if self.g.nodes[key][f"is_{self.SUB_GROUP}_service"] != 1:
                            self.g.nodes[key][f"is_{self.SUB_GROUP}_service"] = 0
                            self.g.nodes[key][f"provision_{self.SUB_GROUP}"] = 0
                            self.g.nodes[key][f"id_{self.SUB_GROUP}"] = 0
                    except KeyError:
                        self.g.nodes[key][f"is_{self.SUB_GROUP}_service"] = 0
                        self.g.nodes[key][f"provision_{self.SUB_GROUP}"] = 0
                        self.g.nodes[key][f"id_{self.SUB_GROUP}"] = 0
                else:
                    self.g.nodes[key][f"is_{self.SUB_GROUP}_service"] = 1
                    self.g.nodes[key][f"{self.SUB_GROUP}_capacity"] = service_blocks_dict[key]
                    self.g.nodes[key][f"provision_{self.SUB_GROUP}"] = 0
                    self.g.nodes[key][f"id_{self.SUB_GROUP}"] = 0

                if self.g.nodes[key]["is_living"] == True:
                    self.g.nodes[key][f"population_prov_{self.SUB_GROUP}"] = 0
                    self.g.nodes[key][f"population_unprov_{self.SUB_GROUP}"] = blocks_geom_dict["population_balanced"][
                        int(key)
                    ]

    def get_stats(self):
        """
        TODO: add docstring
        """
        g = self.g.copy()
        if not g:
            return 0

        blocks_service = 0
        blocks_bad = 0
        blocks_living = 0
        total = 0
        for key in g:
            total += 1
            try:
                if g.nodes[key]["is_living"] == True:
                    blocks_living += 1
                elif g.nodes[key][f"is_{self.SUB_GROUP}_service"] == 1:
                    blocks_service += 1
            except KeyError:
                blocks_bad += 1

        print(f"количество кварталов c сервисом {self.SUB_GROUP}: {blocks_service}")
        print(f"количество жилых кварталов: {blocks_living}")
        print(f"количество кварталов всего: {total}")
        print(f"количество кварталов c ошибкой: {blocks_bad}")

    def get_provision(self):
        SUB_GROUP = self.SUB_GROUP
        g = self.g
        standard = self.standard

        for node in g.nodes:
            if g.nodes[node][f"is_{SUB_GROUP}_service"] == 1:
                neighbors = list(g.neighbors(node))
                capacity = g.nodes[node][f"{SUB_GROUP}_capacity"]
                if (
                    g.nodes[node]["is_living"] == True
                    and g.nodes[node]["population"] > 0
                    and g.nodes[node][f"provision_{self.SUB_GROUP}"] < 100
                ):

                    if g.nodes[node][f"provision_{self.SUB_GROUP}"] == 0:
                        load = (g.nodes[node][f"population_unprov_{SUB_GROUP}"] / 1000) * standard

                    elif g.nodes[node][f"provision_{self.SUB_GROUP}"] > 0:
                        load = (g.nodes[node][f"population_unprov_{SUB_GROUP}"] / 1000) * standard

                    if load <= capacity:
                        capacity -= load
                        g.nodes[node][f"provision_{SUB_GROUP}"] = 100
                        g.nodes[node][f"id_{SUB_GROUP}"] = node
                        g.nodes[node][f"population_prov_{SUB_GROUP}"] += g.nodes[node][f"population_unprov_{SUB_GROUP}"]
                        g.nodes[node][f"population_unprov_{SUB_GROUP}"] -= g.nodes[node][
                            f"population_unprov_{SUB_GROUP}"
                        ]

                    else:
                        if capacity > 0:
                            prov_people = (capacity * 1000) / standard
                            capacity -= capacity

                            g.nodes[node][f"id_{self.SUB_GROUP}"] = node
                            g.nodes[node][f"population_prov_{self.SUB_GROUP}"] += prov_people
                            g.nodes[node][f"population_unprov_{self.SUB_GROUP}"] = (
                                g.nodes[node][f"population_unprov_{self.SUB_GROUP}"] - prov_people
                            )
                            g.nodes[node][f"id_{self.SUB_GROUP}"] = node
                            g.nodes[node][f"provision_{self.SUB_GROUP}"] = (prov_people * 100) / g.nodes[node][
                                "population"
                            ]

                for neighbor in neighbors:
                    if (
                        g.nodes[neighbor]["is_living"] == True
                        and g.nodes[neighbor]["population"] > 0
                        and g.nodes[neighbor][f"is_{SUB_GROUP}_service"] == 0
                        and capacity > 0
                    ):

                        if (
                            g.nodes[neighbor]["is_living"] == True
                            and g.nodes[neighbor]["population"] > 0
                            and g.nodes[neighbor][f"provision_{self.SUB_GROUP}"] < 100
                        ):

                            if g.nodes[neighbor][f"provision_{self.SUB_GROUP}"] == 0:
                                load = (g.nodes[neighbor][f"population_unprov_{SUB_GROUP}"] / 1000) * standard

                            elif g.nodes[neighbor][f"provision_{self.SUB_GROUP}"] > 0:
                                load = (g.nodes[neighbor][f"population_unprov_{SUB_GROUP}"] / 1000) * standard

                            if load <= capacity:
                                capacity -= load
                                g.nodes[neighbor][f"provision_{SUB_GROUP}"] = 100
                                g.nodes[neighbor][f"id_{SUB_GROUP}"] = node
                                g.nodes[neighbor][f"population_prov_{SUB_GROUP}"] += g.nodes[neighbor][
                                    f"population_unprov_{SUB_GROUP}"
                                ]
                                g.nodes[neighbor][f"population_unprov_{SUB_GROUP}"] -= g.nodes[neighbor][
                                    f"population_unprov_{SUB_GROUP}"
                                ]

                            else:
                                if capacity > 0:
                                    prov_people = (capacity * 1000) / standard
                                    capacity -= capacity

                                    g.nodes[neighbor][f"id_{self.SUB_GROUP}"] = neighbor
                                    g.nodes[neighbor][f"population_prov_{self.SUB_GROUP}"] += prov_people
                                    g.nodes[neighbor][f"population_unprov_{self.SUB_GROUP}"] = (
                                        g.nodes[neighbor][f"population_unprov_{self.SUB_GROUP}"] - prov_people
                                    )
                                    g.nodes[neighbor][f"id_{self.SUB_GROUP}"] = node
                                    g.nodes[neighbor][f"provision_{self.SUB_GROUP}"] = (prov_people * 100) / g.nodes[
                                        neighbor
                                    ]["population"]

        self.g = g

    def get_geo(self):
        g = self.g
        blocks = self.blocks.copy()
        blocks[f"provision_{self.SUB_GROUP}"] = 0
        blocks[f"id_{self.SUB_GROUP}"] = 0
        blocks[f"population_prov_{self.SUB_GROUP}"] = 0
        blocks[f"population_unprov_{self.SUB_GROUP}"] = 0
        blocks[f"provision_{self.SUB_GROUP}"] = 0
        blocks["population"] = 0

        for n in g:
            indx = blocks[blocks.index == n].index[0]
            if g.nodes[n]["is_living"] == True:
                if g.nodes[n].get(f"id_{self.SUB_GROUP}") is not None:
                    blocks.loc[indx, f"id_{self.SUB_GROUP}"] = g.nodes[n][f"id_{self.SUB_GROUP}"]
                    blocks.loc[indx, f"population_prov_{self.SUB_GROUP}"] = g.nodes[n][
                        f"population_prov_{self.SUB_GROUP}"
                    ]
                    blocks.loc[indx, f"population_unprov_{self.SUB_GROUP}"] = g.nodes[n][
                        f"population_unprov_{self.SUB_GROUP}"
                    ]
                    blocks.loc[indx, f"provision_{self.SUB_GROUP}"] = g.nodes[n][f"provision_{self.SUB_GROUP}"]
                    blocks.loc[indx, "population"] = g.nodes[n]["population"]

                else:
                    blocks[f"population_unprov_{self.SUB_GROUP}"][indx] = g.nodes[n][
                        f"population_unprov_{self.SUB_GROUP}"
                    ]

        blocks[f"id_{self.SUB_GROUP}"] = blocks[f"id_{self.SUB_GROUP}"].astype(int)
        blocks[f"population_prov_{self.SUB_GROUP}"] = blocks[f"population_prov_{self.SUB_GROUP}"].astype(int)
        blocks[f"population_unprov_{self.SUB_GROUP}"] = blocks[f"population_unprov_{self.SUB_GROUP}"].astype(int)
        blocks[f"provision_{self.SUB_GROUP}"] = blocks[f"provision_{self.SUB_GROUP}"].astype(int)
        blocks["population"] = blocks["population"].astype(int)

        return blocks

    def run(self):
        self.prepare_graph()
        self.get_stats()
        self.get_provision()
        return self.get_geo()
