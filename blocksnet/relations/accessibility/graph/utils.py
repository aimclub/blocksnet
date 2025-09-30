import pandas as pd
import geopandas as gpd
import networkx as nx
from .schemas import validate_accessibility_graph, X_KEY, Y_KEY

GEOMETRY_KEY = "geometry"


def _nodes_to_gdf(graph: nx.Graph, crs):
    """Nodes to gdf.

    Parameters
    ----------
    graph : nx.Graph
        Description.
    crs : Any
        Description.

    """
    data = graph.nodes(data=True)
    df = pd.DataFrame([d[1] for d in data], index=[d[0] for d in data])
    df[GEOMETRY_KEY] = gpd.points_from_xy(df[X_KEY], df[Y_KEY])
    return gpd.GeoDataFrame(df, crs=crs)


def _edges_to_gdf(graph: nx.Graph, crs):
    """Edges to gdf.

    Parameters
    ----------
    graph : nx.Graph
        Description.
    crs : Any
        Description.

    """
    data = graph.edges(data=True)
    index = pd.MultiIndex.from_tuples([(d[0], d[1]) for d in data])
    df = pd.DataFrame([d[2] for d in data], index=index)
    if GEOMETRY_KEY not in df.columns:
        df[GEOMETRY_KEY] = None
    return gpd.GeoDataFrame(df, crs=crs)


def accessibility_graph_to_gdfs(graph: nx.Graph):
    """Accessibility graph to gdfs.

    Parameters
    ----------
    graph : nx.Graph
        Description.

    """
    validate_accessibility_graph(graph)
    crs = graph.graph["crs"]
    nodes_gdf = _nodes_to_gdf(graph, crs)
    edges_gdf = _edges_to_gdf(graph, crs)
    return nodes_gdf, edges_gdf
