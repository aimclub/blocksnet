import shapely
import pandas as pd
import networkx as nx
from pandera import Field
from pandera.typing import Series
from blocksnet.utils.validation import GdfSchema, LandUseSchema


class BlocksSchema(LandUseSchema):
    """Schema for block data required to build origin-destination matrices."""

    population: Series[int] = Field(ge=0)
    site_area: Series[float] = Field(ge=0)


def validate_od_matrix(od_mx: pd.DataFrame, graph: nx.Graph):
    """Validate that an origin-destination matrix matches a network graph.

    Parameters
    ----------
    od_mx : pandas.DataFrame
        Square matrix describing flows between network nodes.
    graph : networkx.Graph
        Graph whose node labels must align with ``od_mx`` indices and columns.

    Raises
    ------
    ValueError
        If the matrix is not a dataframe, is not square, or references nodes
        missing from ``graph``.
    """
    if not isinstance(od_mx, pd.DataFrame):
        raise ValueError("Origin destination matrix must be an instance of pd.DataFrame")
    if not all(od_mx.index == od_mx.columns):
        raise ValueError("Origin destination matrix index and columns must match")
    if not od_mx.index.isin(graph.nodes).all():
        raise ValueError("Origin destination matrix index must be contained in graph nodes labels")
