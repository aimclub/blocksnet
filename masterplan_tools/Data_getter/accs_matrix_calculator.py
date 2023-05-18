"""
TODO: add docstring
"""

import pandas as pd
import numpy as np
import geopandas as gpd
import networkit as nk  # pylint: disable=import-error
import networkx as nx
from tqdm.auto import tqdm

tqdm.pandas()


class Accessibility:
    """
    TODO: add docstring
    """

    def __init__(self, city_crs, blocks, G=None, option="intermodal"):
        self.city_crs = city_crs
        self.blocks = blocks
        self.G = G
        self.option = option

    def get_nx2_nk_idmap(self, G_nx):
        """
        TODO: add docstring
        """

        idmap = dict((id, u) for (id, u) in zip(G_nx.nodes(), range(G_nx.number_of_nodes())))
        return idmap

    def get_nk_attrs(self, G_nx):
        """
        TODO: add docstring
        """

        attrs = dict(
            (u, {"x": d[-1]["x"], "y": d[-1]["y"]})
            for (d, u) in zip(G_nx.nodes(data=True), range(G_nx.number_of_nodes()))
        )
        return attrs

    def convert_nx2nk(self, G_nx, idmap=None, weight=None):
        """
        TODO: add docstring
        """

        if not idmap:
            idmap = self.get_nx2_nk_idmap(G_nx)
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

    def get_nk_distances(self, nk_dists, loc):
        """
        TODO: add docstring
        """

        target_nodes = loc.index
        source_node = loc.name
        distances = [nk_dists.getDistance(source_node, node) for node in target_nodes]

        return pd.Series(data=distances, index=target_nodes)

    def get_matrix(self):
        """
        TODO: add docstring
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
        G_nk = self.convert_nx2nk(G_nx, weight="time_min")

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

        accs_matrix = accs_matrix.apply(lambda x: self.get_nk_distances(nk_dists, x), axis=1)
        accs_matrix.index = self.blocks["id"]
        accs_matrix.columns = self.blocks["id"]

        return accs_matrix
