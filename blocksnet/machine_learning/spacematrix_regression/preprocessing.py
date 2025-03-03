import pandas as pd
import geopandas as gpd
import networkx as nx
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from blocksnet import land_use
from .schemas import BlocksSchema, BlocksLandUseSchema, BlocksIndicatorsSchema
from ..feature_engineering import generate_geometry_features
from ...land_use.enum import LandUse


def split_train_and_test(y: torch.Tensor, train_size: float = 0.8) -> tuple[torch.Tensor, torch.Tensor]:

    train_indices, test_indices = train_test_split(range(len(y)), train_size=train_size)

    train_mask = torch.zeros(len(y), dtype=torch.bool)
    test_mask = torch.zeros(len(y), dtype=torch.bool)

    train_mask[train_indices] = True
    test_mask[test_indices] = True

    return train_mask, test_mask


def _standard_normalize(df: pd.DataFrame) -> pd.DataFrame:
    scaler = StandardScaler()
    data = scaler.fit_transform(df)
    return pd.DataFrame(data, index=df.index, columns=df.columns)


def _land_use_to_one_hot(df: pd.DataFrame) -> pd.DataFrame:
    def _one_hot(land_use):
        one_hot_dict = {lu.value: int(lu.value == land_use) if lu is not None else 0 for lu in list(LandUse)}
        return pd.Series(one_hot_dict)

    return df.land_use.apply(_one_hot)


def features_from_geometries(blocks_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    blocks_gdf = BlocksSchema(blocks_gdf)

    features_df = generate_geometry_features(blocks_gdf, False, False).drop(columns=["geometry"])
    features_df = _standard_normalize(features_df)

    return features_df


def features_from_land_use(blocks_df: pd.DataFrame) -> pd.DataFrame:
    blocks_df = BlocksLandUseSchema(blocks_df)
    features_df = _land_use_to_one_hot(blocks_df)
    return features_df


def initialize_x(geometries_features_df: pd.DataFrame, land_use_features_df: pd.DataFrame) -> torch.Tensor:
    df = geometries_features_df.join(land_use_features_df)
    return torch.tensor(df.values, dtype=torch.float)


def initialize_edge_index(adjacency_graph: nx.Graph) -> torch.Tensor:
    return torch.tensor(list(adjacency_graph.edges), dtype=torch.long).t().contiguous()


def initialize_y(blocks_gdf: gpd.GeoDataFrame) -> torch.Tensor:
    columns = blocks_gdf.columns
    if not ("fsi" in columns and "gsi" in columns and "mxi" in columns):
        raise ValueError("Columns must contain fsi, gsi and mxi.")
    df = BlocksIndicatorsSchema(blocks_gdf)
    return torch.tensor(df.values, dtype=torch.float)


def initialize_data(x, edge_index, y=None) -> Data:
    if y is None:
        return Data(x=x, edge_index=edge_index)
    return Data(x=x, edge_index=edge_index, y=y)


__all__ = [
    "split_train_and_test",
    "features_from_geometries",
    "features_from_land_use",
    "initialize_x",
    "initialize_edge_index",
    "initialize_y",
    "initialize_data",
]
