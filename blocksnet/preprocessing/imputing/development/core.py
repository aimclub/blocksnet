import numpy as np
import torch
import pandas as pd
import networkx as nx
import geopandas as gpd
from sklearn.model_selection import train_test_split
from blocksnet.machine_learning import BaseContext
from blocksnet.relations import validate_adjacency_graph
from .schemas import BlocksSchema, BlocksIndicatorsSchema, BlocksLandUseSchema
from ._strategy import strategy

SITE_AREA_COLUMN = "site_area"
SITE_LENGTH_COLUMN = "site_length"
X_COLUMN = "x"
Y_COLUMN = "y"


class DevelopmentImputer(BaseContext):
    def _preprocess_geometries(self, blocks_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        blocks_gdf = BlocksSchema(blocks_gdf)
        blocks_gdf["site_area"] = blocks_gdf.length
        blocks_gdf["site_length"] = blocks_gdf.length
        blocks_gdf["x"] = blocks_gdf.centroid.x
        blocks_gdf["y"] = blocks_gdf.centroid.y
        return blocks_gdf.drop(columns=["geometry"])

    def _preprocess_land_use(self, blocks_df: pd.DataFrame) -> pd.DataFrame:
        blocks_df = BlocksLandUseSchema(blocks_df)
        return blocks_df

    def _preprocess_indicators(self, blocks_df: pd.DataFrame) -> pd.DataFrame:
        blocks_df = BlocksIndicatorsSchema(blocks_df)
        return blocks_df

    def _preprocess_x(self, blocks_gdf: gpd.GeoDataFrame) -> np.ndarray:
        ...

    def _preprocess_y(self, blocks_gdf: gpd.GeoDataFrame) -> np.ndarray:
        ...

    def _preprocess_gdf(self, blocks_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        geometries_df = self._preprocess_geometries(blocks_gdf)
        land_use_df = self._preprocess_land_use(blocks_gdf)
        indicators_df = self._preprocess_indicators(blocks_gdf)

        df = pd.concat([geometries_df, land_use_df, indicators_df], axis=1)
        df["distance_to_center"] = np.sqrt(df["x"] ** 2 + df["y"] ** 2)
        df["x_normalized"] = df["x"] / (df["x"].std() + 1e-8)
        df["y_normalized"] = df["y"] / (df["y"].std() + 1e-8)
        df["lu_diversity"] = df[BlocksLandUseSchema.columns_()].sum(axis=1)
        df["is_mixed_use"] = (df[BlocksLandUseSchema.columns_()].sum(axis=1) > 1).astype(int)
        df["site_length_log"] = np.log1p(df["site_length"])
        df["site_length_squared"] = df["site_length"] ** 2
        df["build_density"] = df["site_area"] / (df["site_area"] + 1e-8)
        df["living_ratio"] = df["living_area"] / (df["site_area"] + 1e-8)
        df["footprint_ratio"] = df["site_area"] / (df["site_length"] + 1e-8)
        return df

    def _preprocess_edge_index(self, graph: nx.Graph, blocks_gdf: gpd.GeoDataFrame) -> np.ndarray:
        validate_adjacency_graph(graph, blocks_gdf)
        graph = nx.convert_node_labels_to_integers(graph)
        edges_list = list(graph.edges)
        return np.array(edges_list).T

    def _postprocess_y(self, y: np.ndarray, index: list[int]) -> pd.DataFrame:
        ...

    def train(
        self, blocks_gdf: gpd.GeoDataFrame, adjacency_graph: nx.Graph, split_params: dict | None = None
    ) -> tuple[list[float], list[float]]:
        blocks_df = self._preprocess_gdf(blocks_gdf)

        edge_index = self._preprocess_edge_index(adjacency_graph, blocks_gdf)

        split_params = split_params or {"train_size": 0.8, "stratify": y, "random_state": 42}
        x_train, x_test, y_train, y_test = train_test_split(x, y, **split_params)

        train_losses, test_losses = self.strategy.train(x_train, x_test, y_train, y_test)
        # y_pred = self.strategy.predict(x_test)
        return train_losses, test_losses

    def run(self, blocks_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        raise NotImplementedError("Not yet implemented.")

    @classmethod
    def default(cls) -> "DevelopmentImputer":
        return cls(strategy)
