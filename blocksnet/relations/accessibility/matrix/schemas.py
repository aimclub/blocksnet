import shapely
from blocksnet.utils.validation import GdfSchema
import pandas as pd
import numpy as np
from loguru import logger


class BlocksSchema(GdfSchema):
    @classmethod
    def _geometry_types(cls):
        return {shapely.geometry.base.BaseGeometry}


def validate_accessibility_matrix(
    matrix: pd.DataFrame,
    blocks_df: pd.DataFrame | None = None,
    index: bool = True,
    columns: bool = True,
    check_squared: bool = True,
):
    if not isinstance(matrix, pd.DataFrame):
        raise ValueError("Matrix must be an instance of pd.DataFrame")
    if check_squared and not all(matrix.index == matrix.columns):
        raise ValueError("Matrix index and columns must match")

    if blocks_df is not None:
        if not isinstance(blocks_df, pd.DataFrame):
            raise ValueError("Blocks must be provided as pd.DataFrame instance")
        if index and not blocks_df.index.isin(matrix.index).all():
            raise ValueError("Blocks index must be in matrix index")
        if columns and not blocks_df.index.isin(matrix.columns).all():
            raise ValueError("Blocks index must be in matrix columns")

    if matrix.apply(np.isnan).any().any():
        logger.warning("Matrix contains NaN values")

    if matrix.apply(np.isinf).any().any():
        logger.warning("Matrix contains inf values")
