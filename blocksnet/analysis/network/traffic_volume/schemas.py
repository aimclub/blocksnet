import pandas as pd
from pandera import Field
from pandera.typing import Series
from ....utils.validation import LandUseSchema, GdfSchema
import shapely
from loguru import logger
import networkx as nx


class BlocksSchema(GdfSchema, LandUseSchema):
    population: Series[float] = Field(nullable=True)
    density: Series[float] = Field(nullable=True)
    diversity: Series[float] = Field(nullable=True)

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
