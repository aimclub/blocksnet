"""
The module provides all necessary tools to get accesibility matrix from transport graph
"""

import geopandas as gpd
import networkit as nk
import networkx as nx
import pandas as pd
from shapely import Polygon
from tqdm import tqdm
from ..models import BaseSchema
from .graph_generator import GraphGenerator


class BlocksSchema(BaseSchema):
    _geom_types = [Polygon]


class AdjacencyCalculator:  # pylint: disable=too-few-public-methods
    def __init__(self, blocks: gpd.GeoDataFrame, graph: nx.MultiDiGraph):
        blocks = BlocksSchema(blocks)
        assert "crs" in graph.graph, 'Graph should contain "crs" property similar to GeoDataFrame'
        graph = GraphGenerator.validate_graph(graph)
        assert blocks.crs == graph.graph["crs"], "Blocks CRS should match graph CRS"

        self.blocks = blocks
        self.graph = graph

    @staticmethod
    def _get_nx2nk_idmap(graph: nx.Graph) -> dict:  # TODO: add typing for the dict
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

    @staticmethod
    def _get_nk_attrs(graph: nx.Graph) -> dict:  # TODO: add typing for the dict
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

    @classmethod
    def _convert_nx2nk(  # pylint: disable=too-many-locals,invalid-name
        cls, graph_nx: nx.MultiDiGraph, idmap: dict | None = None, weight: str = "weight"
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
            idmap = cls._get_nx2nk_idmap(graph_nx)
        n = max(idmap.values()) + 1
        edges = list(graph_nx.edges())

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

    @staticmethod
    def get_distances(graph, df):
        source_nodes = df.index
        target_nodes = df.columns
        spsp = nk.distance.SPSP(graph, source_nodes, target_nodes)
        spsp.run()
        return {(sn, tn): spsp.getDistance(sn, tn) for sn in source_nodes for tn in target_nodes}

    def run(self, batch_size: int | None = None) -> pd.DataFrame:
        """
        This methods runs graph to matrix calculations

        Returns
        -------
        accs_matrix: pd.DataFrame
            An accessibility matrix that contains time between all blocks in the city
        """

        graph_nx = nx.convert_node_labels_to_integers(self.graph)
        graph_nk = self._convert_nx2nk(graph_nx)

        graph_df = pd.DataFrame.from_dict(dict(graph_nx.nodes(data=True)), orient="index")
        graph_gdf = gpd.GeoDataFrame(
            graph_df, geometry=gpd.points_from_xy(graph_df["x"], graph_df["y"]), crs=self.blocks.crs.to_epsg()
        )

        blocks = self.blocks.copy()
        blocks.geometry = blocks.geometry.representative_point()
        from_blocks = graph_gdf["geometry"].sindex.nearest(blocks["geometry"], return_distance=False, return_all=False)
        accs_matrix = pd.DataFrame(0, index=from_blocks[1], columns=from_blocks[1])

        if batch_size is None:
            nk_dists = nk.distance.SPSP(  # pylint: disable=c-extension-no-member
                graph_nk, sources=accs_matrix.index.values
            ).run()

            accs_matrix = accs_matrix.apply(lambda x: self._get_nk_distances(nk_dists, x), axis=1)
        else:
            k_rows = batch_size
            distances = {}

            for i in tqdm(range(0, len(accs_matrix.index), k_rows)):
                sub_df = accs_matrix.iloc[i : i + k_rows]
                distances.update(self.get_distances(graph_nk, sub_df))

            accs_matrix = accs_matrix.apply(
                lambda x: pd.Series(data=[distances[x.name, i] for i in x.index], index=x.index), axis=1
            )

        accs_matrix.index = blocks.index
        accs_matrix.columns = blocks.index

        # bug fix in city block's closest node is no connecte to actual transport infrastructure
        accs_matrix[accs_matrix > 500] = accs_matrix[accs_matrix < 500].max().max()

        return accs_matrix
