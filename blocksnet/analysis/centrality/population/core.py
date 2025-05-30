import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely import Point
from sklearn.preprocessing import MinMaxScaler

from .schemas import BlocksSchema


DEGREE_CENTRALITY_COLUMN = "degree_centrality"
POPULATION_NORMALIZED_COLUMN = "population_normalized"
DEGREE_CENTRALITY_NORMALIZED_COLUMN = "degree_centrality_normalized"
POPULATION_CENTRALITY_COLUMN = "population_centrality"


def population_centrality(blocks_df: pd.DataFrame, adjacency_graph: nx.Graph) -> pd.DataFrame:
    # get blocks and find neighbors in radius
    blocks_df = BlocksSchema(blocks_df)
    degree_centrality = nx.degree_centrality(adjacency_graph)

    blocks_df[DEGREE_CENTRALITY_COLUMN] = blocks_df.index.map(degree_centrality)

    scaler = MinMaxScaler(feature_range=(1, 2))
    blocks_df.loc[:, [POPULATION_NORMALIZED_COLUMN, DEGREE_CENTRALITY_NORMALIZED_COLUMN]] = scaler.fit_transform(
        blocks_df[["population", DEGREE_CENTRALITY_COLUMN]]
    )

    scaler = MinMaxScaler(feature_range=(0, 1))
    blocks_df[POPULATION_CENTRALITY_COLUMN] = (
        blocks_df[POPULATION_NORMALIZED_COLUMN] * blocks_df[DEGREE_CENTRALITY_NORMALIZED_COLUMN]
    )
    blocks_df[POPULATION_CENTRALITY_COLUMN] = scaler.fit_transform(blocks_df[[POPULATION_CENTRALITY_COLUMN]])

    return blocks_df
