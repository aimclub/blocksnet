import shapely
from blocksnet.utils.validation import GdfSchema
import networkx as nx
import pandas as pd
import pyproj


class TerritorySchema(GdfSchema):
    @classmethod
    def _geometry_types(cls):
        return {shapely.Polygon, shapely.MultiPolygon}


CRS_KEY = "crs"
WEIGHT_KEY = "time_min"
X_KEY = "x"
Y_KEY = "y"


def _validate_crs(graph: nx.Graph):
    if not CRS_KEY in graph.graph:
        raise ValueError(f"Graph should contain {CRS_KEY} data in the form of epsg int value")
    elif not isinstance(graph.graph[CRS_KEY], int):
        raise ValueError(f"{CRS_KEY} must be int")
    else:
        try:
            pyproj.CRS(graph.graph[CRS_KEY])
        except pyproj.exceptions.CRSError:
            raise ValueError(f"Not valid {CRS_KEY} value despite it being int")


def _validate_nodes(graph: nx.Graph):
    xs_and_ys_exist = all(X_KEY in data and Y_KEY in data for _, data in graph.nodes(data=True))
    if not xs_and_ys_exist:
        raise ValueError(f"Node should contain {X_KEY} and {Y_KEY}")


def _validate_edges(graph: nx.Graph, weight_key: str):
    weights_exist = all(weight_key in data for u, v, data in graph.edges(data=True))
    if not weights_exist:
        raise ValueError(f"Edges should contain {weight_key}")


def validate_accessibility_graph(graph: nx.Graph, weight_key: str = WEIGHT_KEY):
    if not isinstance(graph, nx.Graph):
        raise ValueError("Graph must be provided as an instance of nx.Graph")
    _validate_crs(graph)
    _validate_nodes(graph)
    _validate_edges(graph, weight_key)
