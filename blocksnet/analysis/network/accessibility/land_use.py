from typing import Callable
import numpy as np
import pandas as pd
from blocksnet.enums import LandUse
from .basic import _accessibility
from .schemas import LandUseAccessibilityBlocksSchema
from blocksnet.relations import validate_accessibility_matrix

LAND_USE_ACCESSIBILITY_COLUMN = "land_use_accessibility"


def land_use_accessibility(
    accessibility_matrix: pd.DataFrame,
    blocks_df: pd.DataFrame,
    land_use: LandUse,
    out: bool = True,
    agg_func: Callable = np.median,
) -> pd.DataFrame:
    """Land use accessibility.

    Parameters
    ----------
    accessibility_matrix : pd.DataFrame
        Description.
    blocks_df : pd.DataFrame
        Description.
    land_use : LandUse
        Description.
    out : bool, default: True
        Description.
    agg_func : Callable, default: np.median
        Description.

    Returns
    -------
    pd.DataFrame
        Description.

    """
    if not isinstance(land_use, LandUse):
        raise TypeError(f"land_use must be an instance of {LandUse.__name__}")
    blocks_df = LandUseAccessibilityBlocksSchema(blocks_df)
    blocks_idx = blocks_df[blocks_df.land_use == land_use].index
    validate_accessibility_matrix(accessibility_matrix, blocks_df, index=not out, columns=out, check_squared=False)
    acc_mx = accessibility_matrix[blocks_idx] if out else accessibility_matrix.loc[blocks_idx]
    return _accessibility(acc_mx, agg_func=agg_func, out=out, accessibility_column=LAND_USE_ACCESSIBILITY_COLUMN)


def land_use_accessibility_matrix(
    accessibility_matrix: pd.DataFrame, blocks_df: pd.DataFrame, agg_func: Callable = np.median
) -> pd.DataFrame:
    """Land use accessibility matrix.

    Parameters
    ----------
    accessibility_matrix : pd.DataFrame
        Description.
    blocks_df : pd.DataFrame
        Description.
    agg_func : Callable, default: np.median
        Description.

    Returns
    -------
    pd.DataFrame
        Description.

    """
    validate_accessibility_matrix(
        accessibility_matrix, blocks_df=blocks_df, index=True, columns=True, check_squared=False
    )
    lu_mx = pd.DataFrame(0.0, index=list(LandUse), columns=list(LandUse))
    lu_idx = {lu: blocks_df[blocks_df.land_use == lu].index for lu in LandUse}

    for lu_a, idx_a in lu_idx.items():
        for lu_b, idx_b in lu_idx.items():
            acc_mx = accessibility_matrix.loc[idx_a, idx_b]
            lu_mx.loc[lu_a, lu_b] = agg_func(acc_mx.to_numpy())

    return lu_mx
