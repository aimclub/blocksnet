import geopandas as gpd
import pandas as pd
import networkx as nx
from loguru import logger
from .schemas import BlocksSchema, validate_adjacency_graph


def _generate_adjacency_nodes(blocks_gdf: gpd.GeoDataFrame) -> list[int]:
    """Generate adjacency nodes.

    Parameters
    ----------
    blocks_gdf : gpd.GeoDataFrame
        Description.

    Returns
    -------
    list[int]
        Description.

    """
    logger.info("Generating nodes")
    return blocks_gdf.index.to_list()


def _generate_adjacency_edges(blocks_gdf: gpd.GeoDataFrame, buffer_size: int) -> set[tuple[int, int]]:
    """Generate adjacency edges.

    Parameters
    ----------
    blocks_gdf : gpd.GeoDataFrame
        Description.
    buffer_size : int
        Description.

    Returns
    -------
    set[tuple[int, int]]
        Description.

    """
    logger.info("Generating edges")
    blocks_gdf.geometry = blocks_gdf.buffer(buffer_size)
    sjoin_gdf = blocks_gdf.sjoin(blocks_gdf, predicate="intersects")
    sjoin_gdf = sjoin_gdf[sjoin_gdf.index != sjoin_gdf["index_right"]].reset_index()
    pairs = set()
    for _, row in sjoin_gdf.iterrows():
        a, b = row["index"], row["index_right"]
        pairs.add(tuple(sorted((int(a), b))))
    return pairs


def generate_adjacency_graph(blocks_gdf: gpd.GeoDataFrame, buffer_size: int = 0) -> nx.Graph:

    """Generate adjacency graph.

    Parameters
    ----------
    blocks_gdf : gpd.GeoDataFrame
        Description.
    buffer_size : int, default: 0
        Description.

    Returns
    -------
    nx.Graph
        Description.

    """
    blocks_gdf = BlocksSchema(blocks_gdf)

    adj_graph = nx.Graph(None)

    nodes = _generate_adjacency_nodes(blocks_gdf)
    adj_graph.add_nodes_from(nodes)

    edges = _generate_adjacency_edges(blocks_gdf, buffer_size)
    adj_graph.add_edges_from(edges)

    logger.success(
        f"Adjacency graph successfully generated: {len(adj_graph.nodes)} nodes, {len(adj_graph.edges)} edges"
    )
    return adj_graph


def get_adjacency_context(adjacency_graph: nx.Graph, blocks_df: pd.DataFrame, keep: bool = True):
    """Get adjacency context.

    Parameters
    ----------
    adjacency_graph : nx.Graph
        Description.
    blocks_df : pd.DataFrame
        Description.
    keep : bool, default: True
        Description.

    """
    validate_adjacency_graph(adjacency_graph, blocks_df)
    blocks_ids = set(blocks_df.index)
    neighbors = {node for block_id in blocks_ids for node in adjacency_graph.neighbors(block_id)}
    if keep:
        neighbors = neighbors | blocks_ids
    return adjacency_graph.subgraph(neighbors)
