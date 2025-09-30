"""Graph preprocessing helpers for the network classifier."""

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
    """Normalise node coordinate attributes to ``x``/``y`` floats.

    Parameters
    ----------
    node_data : dict of str to Any
        Mutable mapping of node attributes that will be updated in place.

    Returns
    -------
    None

    Raises
    ------
    KeyError
        If neither coordinate key nor its alternative is present in ``node_data``.
    """

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
    """Drop edge metadata, keeping only topology.

    Parameters
    ----------
    edge_data : dict of str to Any
        Edge attributes cleared in place.

    Returns
    -------
    None
    """

    edge_data.clear()


def _preprocess_graph_data(graph_data: dict[str, Any], validate_category: bool):
    """Validate optional settlement category metadata.

    Parameters
    ----------
    graph_data : dict of str to Any
        Graph-level metadata mapping modified in place.
    validate_category : bool
        Whether to enforce the presence of :data:`CATEGORY_KEY`.

    Returns
    -------
    None

    Raises
    ------
    KeyError
        If the category metadata are missing when validation is requested.
    TypeError
        If the category value cannot be coerced to :class:`SettlementCategory`.
    """

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
    """Convert a graph to a simple undirected instance.

    Parameters
    ----------
    graph : networkx.Graph
        Graph possibly containing multiple edges or directionality.

    Returns
    -------
    networkx.Graph
        Simplified graph suitable for subsequent processing.
    """

    graph = graph.copy()
    if isinstance(graph, nx.MultiGraph):
        graph = nx.Graph(graph)
    if graph.is_directed():
        graph = graph.to_undirected()
    return graph


def preprocess_graph(graph: nx.Graph, validate_category: bool) -> nx.Graph:
    """Prepare a raw graph for feature extraction.

    Parameters
    ----------
    graph : networkx.Graph
        Input graph potentially containing redundant metadata.
    validate_category : bool
        Whether to require a valid :class:`SettlementCategory` in the graph metadata.

    Returns
    -------
    networkx.Graph
        Simplified graph with reindexed nodes and cleaned metadata.

    Raises
    ------
    TypeError
        If ``graph`` is not a :class:`networkx.Graph` instance or if metadata are of invalid type.
    KeyError
        If required coordinate or category keys are missing.
    """

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
    """Convert a graph's nodes into a projected GeoDataFrame.

    Parameters
    ----------
    graph : networkx.Graph
        Graph with nodes containing ``x`` and ``y`` coordinates.

    Returns
    -------
    geopandas.GeoDataFrame
        Nodes expressed as geometries in an appropriate projected coordinate reference system.
    """

    nodes_dict = dict(graph.nodes(data=True))
    df = pd.DataFrame.from_dict(nodes_dict, orient="index")
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[X_KEY], df[Y_KEY]), crs=4326)
    return gdf.to_crs(gdf.estimate_utm_crs())
