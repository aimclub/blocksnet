import pandas as pd
from pandera import Field
from pandera.typing import Series
from ....utils.validation import LandUseSchema, GdfSchema
import shapely
from loguru import logger
import networkx as nx
import geopandas as gpd


class BlocksSchema(GdfSchema, LandUseSchema):
    population: Series[float] = Field(default=0)
    density: Series[float] = Field(default=0)
    diversity: Series[float] = Field(default=0)

    @classmethod
    def _geometry_types(cls):
        return {shapely.Polygon}


class NodesSchema(GdfSchema):
    @classmethod
    def _geometry_types(cls):
        return {shapely.Point}


def validate_graph(graph: nx.Graph, param: str):
    if not isinstance(graph, nx.Graph):
        raise ValueError("Graph must be provided as nx.Graph instance")

    for u, v, data in graph.edges(data=True):
        if param not in data:
            raise ValueError(f"Edge ({u}, {v}) is missing required attribute 'time_min'")


def validate_matrix(matrix: pd.DataFrame, from_gdf: gpd.GeoDataFrame, to_gdf: gpd.GeoDataFrame = None):
    if to_gdf is None:
        to_gdf = from_gdf

    if not from_gdf.index.isin(matrix.index).all():
        raise ValueError("Blocks index must be in matrix index")

    if not to_gdf.index.isin(matrix.columns).all():
        raise ValueError("Blocks index must be in matrix index")
