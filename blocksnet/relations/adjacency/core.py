import geopandas as gpd
import networkx as nx
from loguru import logger
from .schemas import BlocksSchema

BUFFER_SIZE = 0


def _generate_adjacency_nodes(blocks_gdf: gpd.GeoDataFrame) -> list[int]:
    logger.info("Generating nodes.")
    return blocks_gdf.index.to_list()


def _generate_adjacency_edges(blocks_gdf: gpd.GeoDataFrame, buffer_size: int) -> list[tuple[int, int]]:
    logger.info("Generating edges.")
    blocks_gdf.geometry = blocks_gdf.buffer(buffer_size)
    sjoin_gdf = blocks_gdf.sjoin(blocks_gdf, predicate="intersects")
    sjoin_gdf = sjoin_gdf[sjoin_gdf.index != sjoin_gdf["index_right"]]
    return [(u, v) for u, v in sjoin_gdf["index_right"].to_dict().items()]


def generate_adjacency_graph(blocks_gdf: gpd.GeoDataFrame, buffer_size: int = BUFFER_SIZE) -> nx.Graph:

    blocks_gdf = BlocksSchema(blocks_gdf)

    adj_graph = nx.Graph(None, crs=blocks_gdf.crs)

    nodes = _generate_adjacency_nodes(blocks_gdf)
    adj_graph.add_nodes_from(nodes)

    edges = _generate_adjacency_edges(blocks_gdf, buffer_size)
    adj_graph.add_edges_from(edges)

    logger.success(
        f"Adjacency graph successfully generated: {len(adj_graph.nodes)} nodes, {len(adj_graph.edges)} edges"
    )
    return adj_graph
