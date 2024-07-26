import geopandas as gpd
import networkx as nx
import numpy as np
from scipy.spatial import cKDTree
from shapely import Point
from sklearn.preprocessing import MinMaxScaler
from ..base_method import BaseMethod


POPULATION_CENTRALITY_COLUMN = "population_centrality"

PLOT_KWARGS = {"column": POPULATION_CENTRALITY_COLUMN, "legend": True, "vmin": 0}


class PopulationCentrality(BaseMethod):
    """
    Provides methods to calculate and visualize the population centrality of urban blocks.
    Population centrality is determined based on connectivity and population density within a specified radius.

    Methods
    -------
    plot(gdf, linewidth=0.1, figsize=(10, 10))
        Plots population centrality data on a map.
    calculate(connectivity_radius=1000) -> gpd.GeoDataFrame
        Calculates population centrality for urban blocks based on connectivity and population density.
    """

    @staticmethod
    def plot(gdf: gpd.GeoDataFrame, linewidth=0.1, figsize: tuple[int, int] = (10, 10)):
        """
        Plots population centrality data for blocks on a map.

        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            GeoDataFrame containing block geometries and population centrality data.
        linewidth : float, optional
            Line width for plotting the geometries, by default 0.1.
        figsize : tuple of int, optional
            Size of the figure to plot, by default (10, 10).

        Returns
        -------
        None
        """
        ax = gdf.plot(color="#ddd", linewidth=linewidth, figsize=figsize)
        gdf.plot(ax=ax, linewidth=linewidth, **PLOT_KWARGS)
        ax.set_axis_off()

    def calculate(self, connectivity_radius: float = 1000) -> gpd.GeoDataFrame:
        """
        Calculate population centrality for urban blocks based on connectivity and population density.

        Parameters
        ----------
        connectivity_radius : float, optional
            The radius within which to consider connectivity between blocks, by default 1000 meters.

        Returns
        -------
        geopandas.GeoDataFrame
            GeoDataFrame of urban blocks with an added column for population centrality.

        This method calculates the connectivity of urban blocks within the specified radius,
        computes the degree centrality of each block, normalizes the values, and combines them with
        normalized population density to determine the final population centrality.
        """
        # get blocks and find neighbors in radius
        blocks = self.city_model.get_blocks_gdf()
        points = np.array([(point.x, point.y) for point in blocks.geometry.centroid])
        tree = cKDTree(points)
        pairs = tree.query_pairs(r=connectivity_radius)
        # graph creation and centrality calc
        connectivity_graph = nx.Graph()
        for idx, point in enumerate(points):
            connectivity_graph.add_node(idx, pos=(point[0], point[1]))
        for i, j in pairs:
            connectivity_graph.add_edge(i, j)
        centrality = nx.degree_centrality(connectivity_graph)

        # connect graph and blocks
        nodes = gpd.GeoDataFrame(
            geometry=[Point(position["pos"]) for _, position in connectivity_graph.nodes(data=True)], crs=blocks.crs
        )
        nodes[POPULATION_CENTRALITY_COLUMN] = centrality
        blocks_out = gpd.sjoin(blocks, nodes, how="left", predicate="intersects")

        # normalize values
        data = blocks_out[POPULATION_CENTRALITY_COLUMN].values.astype(float).reshape(-1, 1)
        minmax_scaler = MinMaxScaler(feature_range=(1, 2))
        normalized_data = minmax_scaler.fit_transform(data)
        blocks_out[POPULATION_CENTRALITY_COLUMN] = normalized_data

        data = blocks_out["population"].values.astype(float).reshape(-1, 1)
        normalized_data = minmax_scaler.fit_transform(data)
        blocks_out["population"] = normalized_data

        minmax_scaler = MinMaxScaler(feature_range=(0, 10))
        blocks_out[POPULATION_CENTRALITY_COLUMN] = np.round(
            minmax_scaler.fit_transform(
                (blocks_out[POPULATION_CENTRALITY_COLUMN] * blocks_out["population"])
                .values.astype(float)
                .reshape(-1, 1)
            ),
            2,
        )

        return blocks_out[["geometry", POPULATION_CENTRALITY_COLUMN]]
