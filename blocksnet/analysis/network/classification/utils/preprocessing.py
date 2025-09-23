from typing import Any
import networkx as nx
import geopandas as gpd
import pandas as pd
from blocksnet.enums import SettlementCategory

X_KEY = "x"
LON_KEY = "lon"
Y_KEY = "y"
LAT_KEY = "lat"
CATEGORY_KEY = "category"

COORDINATES_KEYS_MAPPING = {X_KEY: LON_KEY, Y_KEY: LAT_KEY}


def _preprocess_node_data(node_data: dict[str, Any]):
    data = node_data.copy()
    node_data.clear()
    for key, alt_key in COORDINATES_KEYS_MAPPING.items():
        if key in data:
            value = float(data[key])
        elif alt_key in data:
            value = float(data[alt_key])
        else:
            raise KeyError(f"Neither {key} nor {alt_key} keys were found in node data.")
        node_data[key] = value


def _preprocess_edge_data(edge_data: dict[str, Any]):
    edge_data.clear()


def _preprocess_graph_data(graph_data: dict[str, Any], validate_category: bool):
    data = graph_data.copy()
    graph_data.clear()
    if validate_category:
        if CATEGORY_KEY in data:
            category = data[CATEGORY_KEY]
            if isinstance(category, str) or isinstance(category, SettlementCategory):
                if isinstance(category, str):
                    category = category.lower()
                    category = SettlementCategory(category)
            else:
                raise TypeError(f"Graph {CATEGORY_KEY} must be either str or SettlementCategory.")
            graph_data[CATEGORY_KEY] = category
        else:
            raise KeyError(f"No {CATEGORY_KEY} have been found in .graph data.")


def _simplify_graph(graph: nx.Graph) -> nx.Graph:
    graph = graph.copy()
    if isinstance(graph, nx.MultiGraph):
        graph = nx.Graph(graph)
    if graph.is_directed():
        graph = graph.to_undirected()
    return graph


def preprocess_graph(graph: nx.Graph, validate_category: bool) -> nx.Graph:
    if not isinstance(graph, nx.Graph):
        raise TypeError("Graph should be an instance of nx.Graph.")

    graph = _simplify_graph(graph)

    _preprocess_graph_data(graph.graph, validate_category)
    for _, node_data in graph.nodes(data=True):
        _preprocess_node_data(node_data)
    for _, _, edge_data in graph.edges(data=True):
        _preprocess_edge_data(edge_data)
    return nx.convert_node_labels_to_integers(graph)


def graph_to_gdf(graph: nx.Graph) -> gpd.GeoDataFrame:
    nodes_dict = dict(graph.nodes(data=True))
    df = pd.DataFrame.from_dict(nodes_dict, orient="index")
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[X_KEY], df[Y_KEY]), crs=4326)
    return gdf.to_crs(gdf.estimate_utm_crs())
