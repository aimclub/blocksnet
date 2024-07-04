import geopandas as gpd
import networkx as nx
import numpy as np
from scipy.spatial import cKDTree
from shapely import Point
from sklearn.preprocessing import MinMaxScaler
from ..base_method import BaseMethod


POPULATION_CENTRALITY_COLUMN = "population_centrality"

PLOT_KWARGS = {"column": POPULATION_CENTRALITY_COLUMN, "legend": True, "cmap": "RdYlGn", "vmin": 0}


class PopulationCentrality(BaseMethod):
    """
    This class provides methods to calculate and visualize the population centrality of urban blocks.
    Population centrality is determined based on connectivity and population density within a specified radius.
    """

    @staticmethod
    def plot(gdf: gpd.GeoDataFrame, figsize=(10, 10)):
        ax = gdf.plot(color="#ddd", figsize=(10, 10))
        gdf.plot(ax=ax, **PLOT_KWARGS)
        ax.set_axis_off()

    def calculate(self, connectivity_radius: float = 1000) -> gpd.GeoDataFrame:
        """
        Calculate population centrality for urban blocks based on connectivity and population density.

        Parameters:
        - connectivity_radius: The radius within which to consider connectivity between blocks (default is 1000 meters).

        Returns:
        - A GeoDataFrame of urban blocks with an added column for population centrality.

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
            geometry=[Point(position["pos"]) for _, position in connectivity_graph.nodes(data=True)], crs=32636
        )
        nodes["centrality"] = centrality
        blocks_out = gpd.sjoin(blocks, nodes, how="left", predicate="intersects")

        # normalize values
        data = blocks_out["centrality"].values.astype(float).reshape(-1, 1)
        minmax_scaler = MinMaxScaler(feature_range=(1, 2))
        normalized_data = minmax_scaler.fit_transform(data)
        blocks_out["centrality"] = normalized_data

        data = blocks_out["population"].values.astype(float).reshape(-1, 1)
        normalized_data = minmax_scaler.fit_transform(data)
        blocks_out["population"] = normalized_data

        minmax_scaler = MinMaxScaler(feature_range=(0, 10))
        blocks_out[POPULATION_CENTRALITY_COLUMN] = np.round(
            minmax_scaler.fit_transform(
                (blocks_out["centrality"] * blocks_out["population"]).values.astype(float).reshape(-1, 1)
            ),
            2,
        )

        return blocks_out[["geometry", POPULATION_CENTRALITY_COLUMN]]
