import networkx as nx
import geopandas as gpd
import pandas as pd
from loguru import logger


def ensure_crs(gdf: gpd.GeoDataFrame, *args):
    for arg in args:
        if arg.crs != gdf.crs:
            logger.warning("CRS of GeoDataFrame do not match first provided one. Reprojecting")
            arg.to_crs(gdf.crs, inplace=True)


def validate_matrix(matrix: pd.DataFrame, blocks_df: pd.DataFrame | None = None):
    if not all(matrix.index == matrix.columns):
        raise ValueError("Matrix index and columns must match")
    if blocks_df is not None:
        if not isinstance(blocks_df, pd.DataFrame):
            raise ValueError("Blocks must be provided as pd.DataFrame instance")
        if not blocks_df.index.isin(matrix.index).all():
            raise ValueError("Blocks index must be in matrix index")


def validate_graph(graph: nx.Graph, blocks_df: pd.DataFrame):
    if not isinstance(graph, nx.Graph):
        raise ValueError("Graph must be provided as nx.Graph instance")
    if not isinstance(blocks_df, pd.DataFrame):
        raise ValueError("Blocks must be provided as pd.DataFrame instance")
    if not blocks_df.index.isin(graph.nodes).all():
        raise ValueError("Blocks index must be in graph nodes")
