import geopandas as gpd
import networkx as nx
import iduedu as ie
import pyproj
import pyproj.exceptions
from loguru import logger
from .const import CRS_KEY, WEIGHT_KEY, X_KEY, Y_KEY
from .schemas import BlocksSchema


def _validate_crs(graph: nx.Graph):
    if not CRS_KEY in graph.graph:
        raise ValueError(f"Graph should contain {CRS_KEY} data in the form of epsg int value.")
    elif not isinstance(graph.graph[CRS_KEY], int):
        raise ValueError(f"{CRS_KEY} must be int.")
    else:
        try:
            pyproj.CRS(graph.graph[CRS_KEY])
        except pyproj.exceptions.CRSError:
            raise ValueError(f"Not valid {CRS_KEY} value despite it being int.")


def _validate_nodes(graph: nx.Graph):
    xs_and_ys_exist = all(X_KEY in data and Y_KEY in data for _, data in graph.nodes(data=True))
    if not xs_and_ys_exist:
        raise ValueError(f"Node should contain {X_KEY} and {Y_KEY} to identify locations.")


def _validate_edges(graph: nx.Graph, weight_key: str):
    weights_exist = all(weight_key in data for u, v, data in graph.edges(data=True))
    if not weights_exist:
        raise ValueError(f"Edges should contain {weight_key} to calculate the matrix.")


def _validate_graph(graph: nx.Graph, weight_key: str):
    _validate_crs(graph)
    _validate_nodes(graph)
    _validate_edges(graph, weight_key)


def calculate_accessibility_matrix(
    blocks_gdf: gpd.GeoDataFrame, graph: nx.Graph, weight_key: str = WEIGHT_KEY, *args, **kwargs
):
    _validate_graph(graph, weight_key)
    blocks_gdf = BlocksSchema(blocks_gdf)
    blocks_gdf.geometry = blocks_gdf.representative_point()
    accessibility_matrix = ie.get_adj_matrix_gdf_to_gdf(blocks_gdf, blocks_gdf, graph, weight_key, *args, **kwargs)
    return accessibility_matrix
