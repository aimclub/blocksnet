import geopandas as gpd
import networkx as nx
from ..graph.schemas import validate_accessibility_graph, WEIGHT_KEY
from .schemas import BlocksSchema, validate_accessibility_matrix
import pandas as pd


def calculate_accessibility_matrix(
    blocks_gdf: gpd.GeoDataFrame, graph: nx.Graph, weight_key: str = WEIGHT_KEY, *args, **kwargs
):
    validate_accessibility_graph(graph, weight_key)
    blocks_gdf = BlocksSchema(blocks_gdf)
    blocks_gdf.geometry = blocks_gdf.representative_point()
    import iduedu as ie

    accessibility_matrix = ie.get_adj_matrix_gdf_to_gdf(blocks_gdf, blocks_gdf, graph, weight_key, *args, **kwargs)
    return accessibility_matrix


def get_accessibility_context(
    accessibility_matrix: pd.DataFrame,
    blocks_df: pd.DataFrame,
    accessibility: float,
    out: bool = True,
    keep: bool = True,
) -> pd.DataFrame:
    validate_accessibility_matrix(accessibility_matrix, blocks_df)
    if out:
        accessibility_matrix = accessibility_matrix.transpose()
    acc_mx = accessibility_matrix[blocks_df.index]
    if not keep:
        acc_mx = acc_mx[~acc_mx.index.isin(blocks_df.index)]
    mask = (acc_mx <= accessibility).any(axis=1)
    blocks_ids = list(acc_mx[mask].index)
    return accessibility_matrix.loc[blocks_ids, blocks_ids].copy()
