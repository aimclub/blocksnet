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
    """Summarize accessibility to blocks of a specific land-use category.

    Parameters
    ----------
    accessibility_matrix : pandas.DataFrame
        Accessibility matrix relating origins and destinations.
    blocks_df : pandas.DataFrame
        Block dataframe validated by :class:`LandUseAccessibilityBlocksSchema`.
    land_use : LandUse
        Land-use type identifying which blocks should be considered as
        destinations.
    out : bool, optional
        Direction of aggregation: ``True`` aggregates per origin, ``False`` per
        destination. Default is ``True``.
    agg_func : Callable, optional
        Aggregation function applied to the subset of accessibility values.
        Defaults to ``numpy.median``.

    Returns
    -------
    pandas.DataFrame
        Dataframe with a ``land_use_accessibility`` column for each origin or
        destination.

    Raises
    ------
    TypeError
        If ``land_use`` is not an instance of :class:`blocksnet.enums.LandUse`.
    ValueError
        If validation fails or indices do not align.
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
    """Create a land-use-to-land-use accessibility summary matrix.

    Parameters
    ----------
    accessibility_matrix : pandas.DataFrame
        Accessibility matrix with rows and columns labelled by block
        identifiers.
    blocks_df : pandas.DataFrame
        Block dataframe with land-use annotations used to group rows and
        columns.
    agg_func : Callable, optional
        Reduction applied to each land-use pair submatrix. Defaults to
        ``numpy.median``.

    Returns
    -------
    pandas.DataFrame
        Square dataframe indexed by :class:`LandUse` with aggregated
        accessibility values.

    Raises
    ------
    ValueError
        If input validation fails.
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
