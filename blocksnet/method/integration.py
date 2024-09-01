import shapely
import warnings
import momepy
import geopandas as gpd
import osmnx as ox
import networkx as nx
from sklearn.impute import SimpleImputer
from typing import Literal
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import Optional
from enum import Enum
from pydantic import field_validator, model_validator, ConfigDict
from .base_method import BaseMethod

BLOCKS_GRAPH_FETCH_BUFFER = 1000
BLOCKS_INTERSECTION_BUFFER = -5

INTEGRATION_COLUMN = "integration"
CLUSTER_COLUMN = "cluster"
FSI_COLUMN = "fsi"


class IntegrationType(Enum):
    LOCAL = "local"
    GLOBAL = "global"


class WeightType(Enum):
    ANGULAR = "angle"
    LENGTH = "mm_len"


class Integration(BaseMethod):
    """
    A class for calculating and analyzing spatial integration in city blocks.

    This class extends `BaseMethod` and includes methods for computing integration metrics
    based on spatial networks and clustering city blocks based on their integration characteristics.

    Attributes
    ----------
    graph : Optional[nx.Graph]
        A network graph representing the city's road network. If not provided, it is generated from the city's block geometry.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    graph: Optional[nx.Graph] = None

    @field_validator("graph", mode="before")
    @staticmethod
    def validate_graph(graph):
        """
        Validate and optionally generate the network graph from the city's block geometry.

        Parameters
        ----------
        graph : Optional[nx.Graph]
            The network graph representing the city's road network.

        Returns
        -------
        nx.Graph
            The validated or generated network graph.
        """
        if graph is None:
            polygon = self.city_model.blocks.unary_union.unary_union
            graph = ox.graph_from_polygon(polygon, network_type="drive")
        return graph

    @model_validator(mode="before")
    @staticmethod
    def validate_model(model):
        """
        Validate the city model and generate the network graph if necessary.

        Parameters
        ----------
        model : dict
            A dictionary containing the city model and optionally an existing graph.

        Returns
        -------
        dict
            The validated model dictionary with the graph.
        """
        city = model["city_model"]
        if not "graph" in model:
            blocks_gdf = city.get_blocks_gdf()
            blocks_gdf.geometry = blocks_gdf.buffer(BLOCKS_GRAPH_FETCH_BUFFER)
            unary_union = blocks_gdf.unary_union
            polygon_gdf = gpd.GeoDataFrame([{"geometry": unary_union}], crs=blocks_gdf.crs).to_crs(4326)
            polygon = shapely.make_valid(polygon_gdf.loc[0, "geometry"])
            graph = ox.graph_from_polygon(polygon=polygon, simplify=True, network_type="drive")
            model["graph"] = ox.project_graph(graph, city.crs)
        else:
            graph = model["graph"]
            assert graph.graph["crs"] == city.crs, "Graph CRS should match city CRS"
        return model

    @staticmethod
    def plot(gdf: gpd.GeoDataFrame, column: str, linewidth: float = 0.1, figsize=(10, 10)):
        """
        Plot the GeoDataFrame with a specified column.

        Parameters
        ----------
        gdf : gpd.GeoDataFrame
            The GeoDataFrame to plot.
        column : str
            The column to use for coloring the plot.
        linewidth : float
            Size of polygons' borders, by default 0.1.
        figsize : tuple of int, optional
            The size of the plot figure (default is (10, 10)).

        Returns
        -------
        None
        """
        gdf.plot(
            column=column,
            categorical=True if column == CLUSTER_COLUMN else False,
            legend=True,
            figsize=figsize,
            linewidth=linewidth,
        ).set_axis_off()

    @property
    def blocks(self) -> gpd.GeoDataFrame:
        """
        Get the GeoDataFrame of city blocks with their FSI (Floor Space Index) values.

        Returns
        -------
        gpd.GeoDataFrame
            A GeoDataFrame containing the blocks' geometries and FSI values.
        """
        return self.city_model.get_blocks_gdf()[["geometry", FSI_COLUMN]]

    @property
    def edges(self) -> gpd.GeoDataFrame:
        """
        Get the GeoDataFrame of edges from the network graph.

        Returns
        -------
        gpd.GeoDataFrame
            A GeoDataFrame containing the edges of the network graph.
        """
        _, edges = ox.graph_to_gdfs(self.graph)
        return edges

    def _get_dual_graph(self) -> nx.Graph:
        """
        Generate the dual graph of the city blocks based on their spatial intersections.

        Returns
        -------
        nx.Graph
            The dual graph representing the spatial relationships between city blocks.
        """
        gdf_blocks = self.blocks.copy()
        gdf_blocks.geometry = gdf_blocks.buffer(-5)
        warnings.filterwarnings("ignore", category=FutureWarning)
        merged = gpd.sjoin(self.edges, gdf_blocks, how="left", predicate="intersects")
        graph = merged[merged["index_right"].isna()]
        graph = graph.drop(columns=["index_right"])
        dual_graph = momepy.gdf_to_nx(graph, approach="dual")
        return dual_graph

    def clusterize(
        self, integration_blocks: gpd.GeoDataFrame, columns=[FSI_COLUMN, INTEGRATION_COLUMN], n_clusters=4
    ) -> gpd.GeoDataFrame:
        """
        Perform clustering on integration blocks based on specified columns.

        Parameters
        ----------
        integration_blocks : gpd.GeoDataFrame
            The GeoDataFrame containing blocks with integration metrics.
        columns : list of str, optional
            The columns to use for clustering (default is [FSI_COLUMN, INTEGRATION_COLUMN]).
        n_clusters : int, optional
            The number of clusters to form (default is 4).

        Returns
        -------
        gpd.GeoDataFrame
            The GeoDataFrame with an additional column for cluster labels.
        """

        features = integration_blocks[columns].copy()

        imputer = SimpleImputer(strategy="mean")
        features_imputed = imputer.fit_transform(features)

        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        features_imputed = imputer.fit_transform(features_scaled)

        kmeans = KMeans(n_clusters=n_clusters, n_init=10)
        kmeans.fit(features_imputed)

        blocks = integration_blocks.copy()
        blocks[CLUSTER_COLUMN] = kmeans.labels_

        return blocks

    def calculate(
        self,
        integration_type: IntegrationType | None = None,
        weight_type: WeightType = WeightType.ANGULAR,
        local_radius: int = 5,
    ) -> gpd.GeoDataFrame:
        """
        Calculate integration metrics for city blocks based on the specified integration type and weight type.

        Parameters
        ----------
        integration_type : IntegrationType, optional
            The type of integration to calculate (local or global). If None, no integration is calculated.
        weight_type : WeightType, optional
            The type of weight to use for integration calculation (default is WeightType.ANGULAR).
        local_radius : int, optional
            The radius for local integration calculation (default is 5).

        Returns
        -------
        gpd.GeoDataFrame
            The GeoDataFrame with calculated integration metrics for each block.
        """
        blocks = self.blocks.copy()
        dual_graph = self._get_dual_graph()

        if integration_type == IntegrationType.LOCAL:
            integration = momepy.closeness_centrality(
                dual_graph, radius=local_radius, name=INTEGRATION_COLUMN, weight=weight_type.value
            )
        else:
            integration = momepy.closeness_centrality(
                dual_graph, name=INTEGRATION_COLUMN, verbose=True, weight=weight_type.value
            )

        integration_gdf = momepy.nx_to_gdf(integration, points=False)
        integration_gdf.geometry = integration_gdf.geometry.centroid  # .buffer(5)

        merged = gpd.sjoin_nearest(blocks, integration_gdf, how="left")
        avg_local_integration = merged.groupby("id").agg({INTEGRATION_COLUMN: "mean"})
        blocks[INTEGRATION_COLUMN] = avg_local_integration[INTEGRATION_COLUMN]

        return blocks
