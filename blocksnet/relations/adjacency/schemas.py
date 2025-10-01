import shapely
from blocksnet.utils.validation import GdfSchema
import networkx as nx
import pandas as pd


class BlocksSchema(GdfSchema):
    @classmethod
    def _geometry_types(cls):
        return {shapely.Polygon}


def validate_adjacency_graph(graph: nx.Graph, blocks_df: pd.DataFrame):
    if not isinstance(graph, nx.Graph):
        raise ValueError("Graph must be provided as an instance of nx.Graph")
    if not isinstance(blocks_df, pd.DataFrame):
        raise ValueError("Blocks must be provided as an instance of pd.DataFrame")
    if not blocks_df.index.isin(graph.nodes).all():
        raise ValueError("Blocks index must be in graph nodes")
