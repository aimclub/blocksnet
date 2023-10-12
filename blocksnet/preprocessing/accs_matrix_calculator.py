"""
This module provides all necessary tools to get accesibility matrix from transport graph
"""

import geopandas as gpd
import networkit as nk
import networkx as nx
import pandas as pd
from tqdm import tqdm

tqdm.pandas()


class Accessibility:  # pylint: disable=too-few-public-methods
    """
    Class Accessibility calculates accessibility matrix between city blocks.
    It takes a lot of RAM to calculate one since we have thousands of city blocks.

    Methods
    -------
    get_matrix
    """

    def __init__(self, blocks, graph: nx.Graph = None):
        self.blocks = blocks
        """a dataframe with city blocks"""
        self.G = graph  # pylint: disable=invalid-name
        """transport graph (in networkx format). Walk, drive, bike or transport graph"""

    def _get_nx2nk_idmap(self, graph: nx.Graph) -> dict:  # TODO: add typing for the dict
        """
        This method gets ids from nx graph to place as attribute in nk graph

        Attributes
        ----------
        graph: networkx graph

        Returns
        -------
        idmap: dict
            map of old and new ids
        """

        idmap = dict((id, u) for (id, u) in zip(graph.nodes(), range(graph.number_of_nodes())))
        return idmap

    def _get_nk_attrs(self, graph: nx.Graph) -> dict:  # TODO: add typing for the dict
        """
        This method gets attributes from nx graph to set as attributes in nk graph

        Attributes
        ----------
        graph: networkx graph

        Returns
        -------
        idmap: dict
            map of old and new attributes
        """

        attrs = dict(
            (u, {"x": d[-1]["x"], "y": d[-1]["y"]})
            for (d, u) in zip(graph.nodes(data=True), range(graph.number_of_nodes()))
        )
        return attrs

    def _convert_nx2nk(  # pylint: disable=too-many-locals,invalid-name
        self, graph_nx: nx.Graph, idmap: dict | None = None, weight: str = "time_min"
    ) -> nk.Graph:
        """
        This method converts `networkx` graph to `networkit` graph to fasten calculations.

        Attributes
        ----------
        graph_nx: networkx graph
        idmap: dict
            map of ids in old nx and new nk graphs
        weight: str
            value to be used as a edge's weight

        Returns
        -------
        graph_nk: nk.Graph
            the same graph but now presented in is `networkit` package Graph class.

        """

        if not idmap:
            idmap = self._get_nx2nk_idmap(graph_nx)
        n = max(idmap.values()) + 1
        edges = list(graph_nx.edges())

        if weight:
            graph_nk = nk.Graph(n, directed=graph_nx.is_directed(), weighted=True)
            for u_, v_ in edges:
                u, v = idmap[u_], idmap[v_]
                d = dict(graph_nx[u_][v_])
                if len(d) > 1:
                    for d_ in d.values():
                        v__ = graph_nk.addNodes(2)
                        u__ = v__ - 1
                        w = round(d[weight], 1) if weight in d else 1
                        graph_nk.addEdge(u, v, w)
                        graph_nk.addEdge(u_, u__, 0)
                        graph_nk.addEdge(v_, v__, 0)
                else:
                    d_ = list(d.values())[0]
                    w = round(d_[weight], 1) if weight in d_ else 1
                    graph_nk.addEdge(u, v, w)
        else:
            graph_nk = nk.Graph(n, directed=graph_nx.is_directed())
            for u_, v_ in edges:
                u, v = idmap[u_], idmap[v_]
                graph_nk.addEdge(u, v)

        return graph_nk

    def _get_nk_distances(
        self, nk_dists: nk.base.Algorithm, loc: pd.Series  # pylint: disable=c-extension-no-member
    ) -> pd.Series:
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

    def get_matrix(self, weight="weight") -> pd.DataFrame:
        """
        This methods runs graph to matrix calculations

        Returns
        -------
        accs_matrix: pd.DataFrame
            An accessibility matrix that contains time between all blocks in the city
        """

        graph_nx = nx.convert_node_labels_to_integers(self.G)
        graph_nk = self._convert_nx2nk(graph_nx, weight=weight)

        graph_df = pd.DataFrame.from_dict(dict(graph_nx.nodes(data=True)), orient="index")
        graph_gdf = gpd.GeoDataFrame(
            graph_df, geometry=gpd.points_from_xy(graph_df["x"], graph_df["y"]), crs=self.blocks.crs.to_epsg()
        )

        self.blocks = self.blocks[["geometry"]]
        self.blocks.reset_index(inplace=True)
        self.blocks.rename(columns={"index": "id"}, inplace=True)
        self.blocks["centroids"] = self.blocks["geometry"].representative_point()
        self.blocks.drop(columns=["geometry"], inplace=True)
        self.blocks.rename(columns={"centroids": "geometry"}, inplace=True)

        from_blocks = graph_gdf["geometry"].sindex.nearest(
            self.blocks["geometry"], return_distance=False, return_all=False
        )

        accs_matrix = pd.DataFrame(0, index=from_blocks[1], columns=from_blocks[1])
        nk_dists = nk.distance.SPSP(  # pylint: disable=c-extension-no-member
            graph_nk, sources=accs_matrix.index.values
        ).run()

        accs_matrix = accs_matrix.apply(lambda x: self._get_nk_distances(nk_dists, x), axis=1)
        accs_matrix.index = self.blocks["id"]
        accs_matrix.columns = self.blocks["id"]

        # bug fix in city block's closest node is no connecte to actual transport infrastructure
        accs_matrix[accs_matrix > 500] = accs_matrix[accs_matrix < 500].max().max()

        return accs_matrix
