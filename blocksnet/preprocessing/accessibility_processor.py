"""
IduEdu wrapper. The module is used to generate graph based on user territory and calculate the accessibility matrix.
"""
import geopandas as gpd
import networkx as nx
import osmnx as ox
import pandas as pd
from shapely import MultiPolygon, Polygon
from ..models import BaseSchema
from iduedu import get_adj_matrix_gdf_to_gdf, get_intermodal_graph

IDUEDU_CRS = 4326


class BlocksSchema(BaseSchema):
    """
    Schema class for handling geospatial block data with specified geometry types.

    Attributes
    ----------
    _geom_types : list
        List of allowed geometry types (Polygon, MultiPolygon).
    """

    _geom_types = [Polygon, MultiPolygon]


class AccessibilityProcessor:
    """
    Processor class for calculating accessibility matrix and generating intermodal graph
    for a given set of city blocks.

    Parameters
    ----------
    blocks : gpd.GeoDataFrame
        A GeoDataFrame containing the geospatial data of city blocks.
    """

    def __init__(
        self,
        blocks: gpd.GeoDataFrame,
    ):
        self.blocks = BlocksSchema(blocks)

    @property
    def polygon(self) -> Polygon:
        """
        Computes the convex hull polygon that encloses all blocks.

        Returns
        -------
        shapely.geometry.Polygon
            Convex hull of the blocks in the specified CRS (IDUEDU_CRS).
        """
        return self.blocks.to_crs(IDUEDU_CRS).geometry.unary_union.convex_hull

    @staticmethod
    def plot(blocks: gpd.GeoDataFrame, graph: nx.Graph, figsize: tuple[int, int] = (10, 10), linewidth: float = 0.2):
        """
        Plots the urban blocks and the intermodal city graph on a map.

        Parameters
        ----------
        blocks : gpd.GeoDataFrame
            GeoDataFrame containing the urban blocks geometry.
        graph : nx.Graph
            NetworkX graph representing the intermodal transportation network.
        figsize : tuple[int, int], optional
            Size of the figure (width, height), by default (10, 10).
        linewidth : float, optional
            Width of the plotted edges, by default 0.2.
        """
        ax = blocks.plot(color="#eee", figsize=figsize, linewidth=linewidth)
        _, edges = ox.graph_to_gdfs(graph)
        edges = edges.to_crs(blocks.crs)
        walk_edges = edges[edges["type"] == "walk"]
        walk_edges.plot(ax=ax, color="#ccc", linewidth=linewidth)
        edges = edges[edges["type"] != "walk"]
        edges.plot(ax=ax, column="type", legend=True, linewidth=linewidth)
        ax.set_axis_off()

    @staticmethod
    def _get_broken_nodes(graph: nx.Graph) -> list:
        """
        Identifies broken nodes in the graph that do not have 'x' or 'y' coordinates.

        Parameters
        ----------
        graph : nx.Graph
            NetworkX graph representing the transportation network.

        Returns
        -------
        list
            A list of broken node identifiers.
        """
        return [n for n, data in graph.nodes(data=True) if not "x" in data or not "y" in data]

    @staticmethod
    def _get_island_nodes(graph: nx.Graph) -> list:
        """
        Identifies island nodes in the graph that stay not connected with the main graph.

        Parameters
        ----------
        graph : nx.Graph
            NetworkX graph representing the transportation network.

        Returns
        -------
        list
            A list of island node identifiers.
        """
        components = sorted(nx.strongly_connected_components(graph), key=len)
        components = list(components)[:-1]
        return [node for comp in components for node in comp]

    @classmethod
    def _fix_graph(cls, graph) -> None:
        """
        Removes broken nodes that do not have valid coordinates and islands
        from the graph inplace.

        Parameters
        ----------
        graph : nx.Graph
            NetworkX graph representing the transportation network.
        """
        broken_nodes = cls._get_broken_nodes(graph)
        graph.remove_nodes_from(broken_nodes)
        island_nodes = cls._get_island_nodes(graph)
        graph.remove_nodes_from(island_nodes)

    def get_intermodal_graph(self, clip_by_bounds: bool = True, keep_routes_geom: bool = True) -> nx.Graph:
        """
        Generates an intermodal transportation graph for the given blocks area.

        Parameters
        ----------
        clip_by_bounds : bool, optional
            Whether to clip the graph by the polygon bounds, by default True.
        keep_routes_geom : bool, optional
            Whether to keep the geometry of the routes in the graph, by default True.

        Returns
        -------
        nx.Graph
            NetworkX graph representing the intermodal transportation network.
        """
        graph = get_intermodal_graph(
            polygon=self.polygon, clip_by_bounds=clip_by_bounds, keep_routes_geom=keep_routes_geom
        )
        self._fix_graph(graph)
        return graph

    def get_accessibility_matrix(self, graph: nx.Graph, weight: str = "time_min") -> pd.DataFrame:
        """
        Calculates the accessibility matrix between city blocks using the provided graph.

        Parameters
        ----------
        graph : nx.Graph
            NetworkX graph representing the intermodal transportation network.
        weight : str, optional
            Edge attribute to use for calculating distances (e.g., 'time_min'), by default "time_min".

        Returns
        -------
        pd.DataFrame
            Accessibility matrix as a DataFrame.
        """
        gdf = self.blocks.copy()
        gdf.geometry = gdf.representative_point()
        acc_mx = get_adj_matrix_gdf_to_gdf(gdf, gdf, graph, weight)
        return acc_mx
