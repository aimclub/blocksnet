import shapely
import pandas as pd
import networkx as nx
from pandera import Field
from pandera.typing import Series
from blocksnet.utils.validation import GdfSchema, LandUseSchema


class BlocksSchema(LandUseSchema):
    population: Series[int] = Field(ge=0)
    site_area: Series[float] = Field(ge=0)


def validate_od_matrix(od_mx: pd.DataFrame, graph: nx.Graph):
    if not isinstance(od_mx, pd.DataFrame):
        raise ValueError("Origin destination matrix must be an instance of pd.DataFrame")
    if not all(od_mx.index == od_mx.columns):
        raise ValueError("Origin destination matrix index and columns must match")
    if not od_mx.index.isin(graph.nodes).all():
        raise ValueError("Origin destination matrix index must be contained in graph nodes labels")
