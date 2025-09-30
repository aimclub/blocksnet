import geopandas as gpd
import pandas as pd
import networkx as nx
from loguru import logger
from .schemas import BlocksSchema, validate_adjacency_graph


def _generate_adjacency_nodes(blocks_gdf: gpd.GeoDataFrame) -> list[int]:
    logger.info("Generating nodes")
    return blocks_gdf.index.to_list()


def _generate_adjacency_edges(blocks_gdf: gpd.GeoDataFrame, buffer_size: int) -> set[tuple[int, int]]:
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
    """Create a graph describing spatial adjacency between blocks.

    Parameters
    ----------
    blocks_gdf : geopandas.GeoDataFrame
        GeoDataFrame of block geometries validated by :class:`BlocksSchema`.
    buffer_size : int, default=0
        Buffer size in CRS units applied before computing spatial joins. A
        positive buffer enlarges geometries to include near-misses.

    Returns
    -------
    networkx.Graph
        Graph with block indices as nodes and adjacency relations as edges.
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
    """Extract nodes adjacent to a set of blocks.

    Parameters
    ----------
    adjacency_graph : networkx.Graph
        Spatial adjacency graph validated by :func:`validate_adjacency_graph`.
    blocks_df : pandas.DataFrame
        DataFrame whose index contains the focal block identifiers.
    keep : bool, default=True
        If ``True``, include the original blocks in the resulting subgraph.

    Returns
    -------
    networkx.Graph
        Subgraph containing neighbouring block nodes.
    """

    validate_adjacency_graph(adjacency_graph, blocks_df)
    blocks_ids = set(blocks_df.index)
    neighbors = {node for block_id in blocks_ids for node in adjacency_graph.neighbors(block_id)}
    if keep:
        neighbors = neighbors | blocks_ids
    return adjacency_graph.subgraph(neighbors)
