import shapely
from blocksnet.utils.validation import GdfSchema
import pandas as pd
import numpy as np
from loguru import logger


class BlocksSchema(GdfSchema):
    @classmethod
    def _geometry_types(cls):
        return {shapely.geometry.base.BaseGeometry}


def validate_accessibility_matrix(matrix: pd.DataFrame, blocks_df: pd.DataFrame | None = None):
    if not isinstance(matrix, pd.DataFrame):
        raise ValueError("Matrix must be an instance of pd.DataFrame")
    if not all(matrix.index == matrix.columns):
        raise ValueError("Matrix index and columns must match")

    if blocks_df is not None:
        if not isinstance(blocks_df, pd.DataFrame):
            raise ValueError("Blocks must be provided as pd.DataFrame instance")
        if not blocks_df.index.isin(matrix.index).all():
            raise ValueError("Blocks index must be in matrix index")

    if matrix.apply(np.isnan).any().any():
        logger.warning("Matrix contains NaN values")

    if matrix.apply(np.isinf).any().any():
        logger.warning("Matrix contains inf values")
