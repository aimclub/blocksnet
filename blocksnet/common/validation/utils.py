import networkx as nx
import pandas as pd


def validate_matrix(matrix: pd.DataFrame, blocks_df: pd.DataFrame | None = None):
    if not all(matrix.index == matrix.columns):
        raise ValueError("Matrix index and columns must match")
    if blocks_df is not None:
        if not blocks_df.index.isin(matrix.index).all():
            raise ValueError("Block index must be in matrix index")
