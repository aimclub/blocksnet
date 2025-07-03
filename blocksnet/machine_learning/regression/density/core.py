import torch
from torch_geometric.data import Data
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import pandas as pd
import geopandas as gpd
import networkx as nx
from loguru import logger
from .common import SageModel, ModelWrapper, ScalerWrapper
from .schemas import BlocksSchema, BlocksGeometriesSchema, BlocksLandUseSchema, BlocksDensitiesSchema
from blocksnet.preprocessing.feature_engineering import generate_geometries_features
from blocksnet.relations import validate_adjacency_graph

from pathlib import Path

CURRENT_DIRECTORY = Path(__file__).parent
MODELS_DIRECTORY = CURRENT_DIRECTORY / "models"
MODEL_PATH = str(MODELS_DIRECTORY / "model.pt")
SCALER_PATH = str(MODELS_DIRECTORY / "scaler.pkl")


class DensityRegressor(ModelWrapper, ScalerWrapper):
    def __init__(
        self,
        model_class: type[torch.nn.Module] = SageModel,
        model_path: str = MODEL_PATH,
        scaler_path: str = SCALER_PATH,
        *args,
        **kwargs,
    ):
        n_features = len(BlocksLandUseSchema._columns()) + len(BlocksGeometriesSchema._columns())
        n_targets = len(BlocksDensitiesSchema._columns())
        ModelWrapper.__init__(self, n_features, n_targets, model_class, model_path, *args, **kwargs)
        ScalerWrapper.__init__(self, scaler_path)

    def _features_from_land_use(self, blocks_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        df = BlocksLandUseSchema(blocks_gdf)
        return df

    def _features_from_geometries(self, blocks_gdf: gpd.GeoDataFrame, fit_scaler: bool) -> pd.DataFrame:
        gdf = BlocksSchema(blocks_gdf)
        gdf = generate_geometries_features(
            gdf, radiuses=False, aspect_ratios=False, centerlines=False, combinations=False
        )
        df = BlocksGeometriesSchema(gdf)
        if fit_scaler:
            logger.info("Fitting the scaler")
            self.scaler.fit(df)
        data = self.scaler.fit_transform(df)
        return pd.DataFrame(data, index=df.index, columns=df.columns)

    def _initialize_x(self, blocks_gdf: gpd.GeoDataFrame, fit_scaler: bool) -> torch.Tensor:
        land_use_df = self._features_from_land_use(blocks_gdf)
        geometries_df = self._features_from_geometries(blocks_gdf, fit_scaler)
        df = land_use_df.join(geometries_df)
        return torch.tensor(df.values, dtype=torch.float)

    def _initialize_edge_index(self, adjacency_graph: nx.Graph) -> torch.Tensor:
        adjacency_graph = nx.convert_node_labels_to_integers(adjacency_graph)
        edges_list = list(adjacency_graph.edges)
        edges_tensor = torch.tensor(edges_list, dtype=torch.long)
        return edges_tensor.t().contiguous()

    def _initialize_y(self, blocks_gdf: gpd.GeoDataFrame) -> torch.Tensor:
        df = BlocksDensitiesSchema(blocks_gdf)
        return torch.tensor(df.values, dtype=torch.float)

    def get_train_data(
        self,
        blocks_gdf: gpd.GeoDataFrame,
        adjacency_graph: nx.Graph,
        fit_scaler: bool = True,
        test: float | list[int] = 0.2,
    ) -> Data:
        validate_adjacency_graph(adjacency_graph, blocks_gdf)
        x = self._initialize_x(blocks_gdf, fit_scaler)
        edge_index = self._initialize_edge_index(adjacency_graph)
        y = self._initialize_y(blocks_gdf)

        train_mask = torch.zeros(len(y), dtype=torch.bool)
        test_mask = torch.zeros(len(y), dtype=torch.bool)

        if isinstance(test, float):
            train_indices, test_indices = train_test_split(range(len(blocks_gdf)), test_size=test)
        else:
            train_index = blocks_gdf[~blocks_gdf.index.isin(test)].index
            test_index = blocks_gdf[blocks_gdf.index.isin(test)].index
            train_indices = blocks_gdf.index.get_indexer(train_index)
            test_indices = blocks_gdf.index.get_indexer(test_index)

        train_mask[train_indices] = True
        test_mask[test_indices] = True

        return Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, test_mask=test_mask)

    def train(
        self,
        data,
        epochs: int = 1_000,
        learning_rate: float = 3e-4,
        weight_decay: float = 5e-4,
        loss_fn=F.huber_loss,
        **kwargs,
    ):
        return self._train_model(
            data, epochs=epochs, learning_rate=learning_rate, weight_decay=weight_decay, loss_fn=loss_fn, **kwargs
        )

    def test(self, data, loss_fn=F.huber_loss, **kwargs):
        return self._test_model(data, loss_fn=loss_fn, **kwargs)

    def evaluate(self, blocks_gdf: gpd.GeoDataFrame, adjacency_graph: nx.Graph) -> gpd.GeoDataFrame:
        validate_adjacency_graph(adjacency_graph, blocks_gdf)

        x = self._initialize_x(blocks_gdf, fit_scaler=False)
        edge_index = self._initialize_edge_index(adjacency_graph)
        data = Data(x=x, edge_index=edge_index)

        out = self._evaluate_model(data)
        data = out.detach().numpy()
        return pd.DataFrame(data, index=blocks_gdf.index, columns=BlocksDensitiesSchema._columns())
