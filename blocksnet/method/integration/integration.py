import shapely
import warnings
import momepy
import geopandas as gpd
import osmnx as ox
import networkx as nx
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import Optional
from enum import Enum
from pydantic import field_validator, model_validator, ConfigDict
from ..base_method import BaseMethod

BLOCKS_GRAPH_FETCH_BUFFER = 1000
BLOCKS_INTERSECTION_BUFFER = -5


class IntegrationType(Enum):
    LOCAL = "local"
    GLOBAL = "global"


class WeightType(Enum):
    ANGULAR = "angle"
    LENGTH = "mm_len"


class Integration(BaseMethod):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    graph: Optional[nx.Graph] = None

    @field_validator("graph", mode="before")
    @staticmethod
    def validate_graph(graph):
        if graph is None:
            polygon = self.city_model.blocks.unary_union.unary_union
            graph = ox.graph_from_polygon(polygon, network_type="drive")
        return graph

    @model_validator(mode="before")
    @staticmethod
    def validate_model(model):
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

    @property
    def blocks(self):
        return self.city_model.get_blocks_gdf()[["geometry", "fsi"]]

    @property
    def edges(self):
        _, edges = ox.graph_to_gdfs(self.graph)
        return edges

    def _get_dual_graph(self):
        gdf_blocks = self.blocks.copy()
        gdf_blocks.geometry = gdf_blocks.buffer(-5)
        warnings.filterwarnings("ignore", category=FutureWarning)
        merged = gpd.sjoin(self.edges, gdf_blocks, how="left", predicate="intersects")
        graph = merged[merged["index_right"].isna()]
        graph = graph.drop(columns=["index_right"])
        dual_graph = momepy.gdf_to_nx(graph, approach="dual")
        return dual_graph

    def clusterize(self, integration_blocks: gpd.GeoDataFrame, columns=["fsi", "integration"], n_clusters=4):

        features = integration_blocks[columns].copy()

        imputer = SimpleImputer(strategy="mean")
        features_imputed = imputer.fit_transform(features)

        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        features_imputed = imputer.fit_transform(features_scaled)

        kmeans = KMeans(n_clusters=n_clusters, n_init=10)
        kmeans.fit(features_imputed)

        blocks = integration_blocks.copy()
        blocks["cluster"] = kmeans.labels_

        return blocks

    def calculate(
        self,
        integration_type: IntegrationType | None = None,
        weight_type: WeightType = WeightType.ANGULAR,
        local_radius: int = 5,
        use_networkit=True,
    ):
        blocks = self.blocks.copy()
        dual_graph = self._get_dual_graph()

        if integration_type == IntegrationType.LOCAL:
            integration = momepy.closeness_centrality(
                dual_graph, radius=local_radius, name="integration", weight=weight_type.value
            )
        if integration_type == IntegrationType.GLOBAL:
            integration = momepy.closeness_centrality(
                dual_graph, name="integration", verbose=True, weight=weight_type.value
            )

        integration_gdf = momepy.nx_to_gdf(integration, points=False)
        integration_gdf.geometry = integration_gdf.geometry.centroid  # .buffer(5)

        merged = gpd.sjoin_nearest(blocks, integration_gdf, how="left")
        avg_local_integration = merged.groupby("id").agg({"integration": "mean"})
        blocks["integration"] = avg_local_integration["integration"]

        return blocks
