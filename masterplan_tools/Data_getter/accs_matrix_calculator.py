"""
This module provides all necessary tools to get accesibility matrix from transport graph
"""

import pandas as pd
import numpy as np  # pylint: disable=import-error
import geopandas as gpd  # pylint: disable=import-error
import networkit as nk  # pylint: disable=import-error
import networkx as nx
from tqdm.auto import tqdm  # pylint: disable=import-error

tqdm.pandas()


class Accessibility:
    """
    Class Accessibility calculates accessibility matrix between city blocks.
    It takes a lot of RAM to calculate one since we have thousands of city blocks.

    Methods
    -------
    get_matrix
    """

    def __init__(self, city_crs: int, blocks, G: nx.Graph = None, option: str = "intermodal"):
        self.city_crs = city_crs
        """city crs"""
        self.blocks = blocks
        """a dataframe with city blocks"""
        self.G = G
        """transport graph (in networkx format). Walk, drive, bike or transport graph"""
        self.option = option
        """type of transport"""

    def _get_nx2_nk_idmap(self, G_nx: nx.Graph) -> dict:
        """
        This method gets ids from nx graph to place as attribute in nk graph

        Attributes
        ----------
        G_nx: networkx graph

        Returns
        -------
        idmap: dict
            map of old and new ids
        """

        idmap = dict((id, u) for (id, u) in zip(G_nx.nodes(), range(G_nx.number_of_nodes())))
        return idmap

    def _get_nk_attrs(self, G_nx: nx.Graph) -> dict:
        """
        This method gets attributes from nx graph to set as attributes in nk graph

        Attributes
        ----------
        G_nx: networkx graph

        Returns
        -------
        idmap: dict
            map of old and new attributes
        """

        attrs = dict(
            (u, {"x": d[-1]["x"], "y": d[-1]["y"]})
            for (d, u) in zip(G_nx.nodes(data=True), range(G_nx.number_of_nodes()))
        )
        return attrs

    def _convert_nx2nk(self, G_nx: nx.Graph, idmap: dict = dict, weight: str = "") -> nk.Graph:
        """
        This method converts nx graph to nk graph to fasten calculations.

        Attributes
        ----------
        G_nx: networkx graph
        idmap: dict
            map of ids in old nx and new nk graphs
        weight: str
            value to be used as a edge's weight

        Returns
        -------
        G_nk: nk.Graph
            the same graph but now it is nk graph not nx one

        """

        if not idmap:
            idmap = self._get_nx2_nk_idmap(G_nx)
        n = max(idmap.values()) + 1
        edges = list(G_nx.edges())

        if weight:
            G_nk = nk.Graph(n, directed=G_nx.is_directed(), weighted=True)
            for u_, v_ in tqdm(edges):
                u, v = idmap[u_], idmap[v_]
                d = dict(G_nx[u_][v_])
                if len(d) > 1:
                    for __ in d.values():
                        v__ = G_nk.addNodes(2)
                        u__ = v__ - 1
                        w = round(d_[weight], 1) if weight in d_ else 1
                        G_nk.addEdge(u, v, w)
                        G_nk.addEdge(u_, u__, 0)
                        G_nk.addEdge(v_, v__, 0)
                else:
                    d_ = list(d.values())[0]
                    w = round(d_[weight], 1) if weight in d_ else 1
                    G_nk.addEdge(u, v, w)
        else:
            G_nk = nk.Graph(n, directed=G_nx.is_directed())
            for u_, v_ in edges:
                u, v = idmap[u_], idmap[v_]
                G_nk.addEdge(u, v)

        return G_nk

    def _get_nk_distances(self, nk_dists: nk.base.Algorithm, loc: pd.Series) -> pd.Series:
        """
        This method calculates distances between blocks using nk SPSP algorithm.
        The function is called inside apply function.

        Attributes
        ----------
        nk_dists: nk.base.Algorithm
            Compressed nk graph to compute distances between nodes using SPSP algorithm
        loc: pd.Series
            Row in the df

        Returns
        -------
        pd.Series with computed distances
        """

        target_nodes = loc.index
        source_node = loc.name
        distances = [nk_dists.getDistance(source_node, node) for node in target_nodes]

        return pd.Series(data=distances, index=target_nodes)

    def get_matrix(self) -> pd.DataFrame:
        """
        This methods runs graph to matrix calculations

        Returns
        -------
        accs_matrix: pd.DataFrame
            An accessibility matrix that contains time between all blocks in the city
        """

        transportation_types = {
            "pedestrian": ["walk"],
            "transport": ["subway", "bus", "trolleybus", "tram"],
            "intermodal": ["subway", "bus", "trolleybus", "tram", "walk"],
            "drive": ["car"],
        }

        ebunch = list(
            ((u, v) for u, v, e in self.G.edges(data=True) if e["type"] not in [transportation_types[self.option]])
        )
        self.G.remove_edges_from(ebunch)

        G_nx = nx.convert_node_labels_to_integers(self.G)
        G_nk = self._convert_nx2nk(G_nx, weight="time_min")

        self.G = None
        G_nx = None

        graph_df = pd.DataFrame.from_dict(dict(G_nx.nodes(data=True)), orient="index")
        graph_gdf = gpd.GeoDataFrame(
            graph_df, geometry=gpd.points_from_xy(graph_df["x"], graph_df["y"]), crs=self.city_crs
        )

        self.blocks = self.blocks[["geometry"]]
        self.blocks.reset_index(inplace=True)
        self.blocks.rename(columns={"index": "id"}, inplace=True)
        self.blocks["centroids"] = self.blocks["geometry"].centroid
        self.blocks.drop(columns=["geometry"], inplace=True)
        self.blocks.rename(columns={"centroids": "geometry"}, inplace=True)

        from_blocks = graph_gdf["geometry"].sindex.nearest(
            self.blocks["geometry"], return_distance=False, return_all=False
        )

        accs_matrix = pd.DataFrame(0, index=from_blocks[1], columns=from_blocks[1])
        nk_dists = nk.distance.SPSP(G_nk, sources=accs_matrix.index.values).run()

        accs_matrix = accs_matrix.apply(lambda x: self._get_nk_distances(nk_dists, x), axis=1)
        accs_matrix.index = self.blocks["id"]
        accs_matrix.columns = self.blocks["id"]

        return accs_matrix
