import numpy as np
import torch
import pandas as pd
import networkx as nx
import geopandas as gpd
from sklearn.model_selection import train_test_split
from blocksnet.machine_learning import BaseContext
from blocksnet.relations import validate_adjacency_graph
from .schemas import BlocksSchema, BlocksIndicatorsSchema, BlocksLandUseSchema
from ._strategy import get_default_strategy
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

SITE_AREA_COLUMN = "site_area"
SITE_LENGTH_COLUMN = "site_length"
X_COLUMN = "x"
Y_COLUMN = "y"


class DevelopmentImputer(BaseContext):
    def _preprocess_geometries(self, blocks_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        blocks_gdf = BlocksSchema(blocks_gdf)
        blocks_gdf["site_area"] = blocks_gdf.length
        blocks_gdf["site_length"] = blocks_gdf.length
        centroid = blocks_gdf.buffer(0).union_all().centroid
        blocks_gdf["x"] = blocks_gdf.centroid.x - centroid.x
        blocks_gdf["y"] = blocks_gdf.centroid.y - centroid.y
        return blocks_gdf.drop(columns=["geometry"])

    def _preprocess_land_use(self, blocks_df: pd.DataFrame) -> pd.DataFrame:
        blocks_df = BlocksLandUseSchema(blocks_df)
        return blocks_df

    def _preprocess_indicators(self, blocks_df: pd.DataFrame) -> pd.DataFrame:
        blocks_df = BlocksIndicatorsSchema(blocks_df)
        return blocks_df

    def _preprocess_x(self, blocks_gdf: gpd.GeoDataFrame) -> np.ndarray:
        geometries_df = self._preprocess_geometries(blocks_gdf)
        land_use_df = self._preprocess_land_use(blocks_gdf)
        df = pd.concat([geometries_df, land_use_df], axis=1)
        return df.values

    def _preprocess_y(self, blocks_df: pd.DataFrame) -> np.ndarray:
        indicators_df = self._preprocess_indicators(blocks_df)
        return indicators_df.values

    def _preprocess_edge_index(self, graph: nx.Graph, blocks_gdf: gpd.GeoDataFrame) -> np.ndarray:
        validate_adjacency_graph(graph, blocks_gdf)
        graph = nx.convert_node_labels_to_integers(graph)
        edges_list = list(graph.edges)
        return np.array(edges_list).T

    def _postprocess_y(self, y: np.ndarray, index: list[int]) -> pd.DataFrame:
        return pd.DataFrame(y, index=index, columns=BlocksIndicatorsSchema.columns_())

    def _split_data(self, x: np.ndarray, split_params: dict) -> tuple[np.ndarray, np.ndarray]:
        size = len(x)

        train_mask = np.zeros(size, dtype=bool)
        test_mask = np.zeros(size, dtype=bool)

        train_indices, test_indices = train_test_split(range(size), **split_params)

        train_mask[train_indices] = True
        test_mask[test_indices] = True
        return train_mask, test_mask

    def train(
        self,
        blocks_gdf: gpd.GeoDataFrame,
        adjacency_graph: nx.Graph,
        split_params: dict | None = None,
        train_params: dict | None = None,
    ) -> tuple[list[float], list[float]]:

        x = self._preprocess_x(blocks_gdf)
        y = self._preprocess_y(blocks_gdf)
        edge_index = self._preprocess_edge_index(adjacency_graph, blocks_gdf)

        split_params = split_params or {"train_size": 0.8, "random_state": 42}
        train_mask, test_mask = self._split_data(x, split_params)

        train_params = train_params or {
            "epochs": 1000,
            "optimizer_params": {"lr": 1e-4, "weight_decay": 1e-3},
        }
        train_losses, test_losses = self.strategy.train(
            x=x,
            y=y,
            edge_index=edge_index,
            train_mask=train_mask,
            test_mask=test_mask,
            **train_params,
        )
        return train_losses, test_losses

    def validate(
        self, blocks_gdf: gpd.GeoDataFrame, adjacency_graph: nx.Graph, split_params: dict | None = None
    ) -> dict[str, float]:
        x = self._preprocess_x(blocks_gdf)
        y = self._preprocess_y(blocks_gdf)
        edge_index = self._preprocess_edge_index(adjacency_graph, blocks_gdf)
        split_params = split_params or {"train_size": 0.8, "random_state": 42}
        _, test_mask = self._split_data(x, split_params)
        y_pred = self.strategy.predict(
            x=x,
            y=y,
            edge_index=edge_index,
            imputation_mask=test_mask,
        )
        df = self._postprocess_y(y_pred, list(blocks_gdf.index))

        metrics = {}
        test_indices = blocks_gdf.index[test_mask]

        for col in df.columns:
            y_true = blocks_gdf.loc[test_indices, col]
            y_hat = df.loc[test_indices, col]

            metrics[col] = {
                "mae": mean_absolute_error(y_true, y_hat),
                "mse": mean_squared_error(y_true, y_hat),
                "r2": r2_score(y_true, y_hat),
            }

        return pd.DataFrame.from_dict(metrics, orient="index")

    def run(self, blocks_gdf: gpd.GeoDataFrame, adjacency_graph: nx.Graph, blocks_ids: list[int]) -> pd.DataFrame:
        x = self._preprocess_x(blocks_gdf)
        y = self._preprocess_y(blocks_gdf)
        edge_index = self._preprocess_edge_index(adjacency_graph, blocks_gdf)

        imputation_mask = np.zeros(len(blocks_gdf), dtype=bool)
        imputation_mask[blocks_gdf.index.get_indexer(blocks_ids)] = True

        y_pred = self.strategy.predict(
            x=x,
            y=y,
            edge_index=edge_index,
            imputation_mask=imputation_mask,
        )

        return self._postprocess_y(y_pred, list(blocks_gdf.index))

    @classmethod
    def default(cls) -> "DevelopmentImputer":
        return cls(get_default_strategy())
