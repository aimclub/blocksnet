"""Generic accessibility aggregation helpers."""

from typing import Callable

import numpy as np
import pandas as pd

from blocksnet.relations import validate_accessibility_matrix

ACCESSIBILITY_COLUMN = "accessibility"
MEDIAN_ACCESSIBILITY_COLUMN = "median_accessibility"
MEAN_ACCESSIBILITY_COLUMN = "mean_accessibility"
MAX_ACCESSIBILITY_COLUMN = "max_accessibility"


def _accessibility(
    accessibility_matrix: pd.DataFrame, agg_func: Callable, out: bool, accessibility_column: str
) -> pd.DataFrame:
    """Aggregate an accessibility matrix with a custom reducer.

    Parameters
    ----------
    accessibility_matrix : pandas.DataFrame
        Accessibility weights between origins (rows) and destinations (columns).
    agg_func : Callable
        Callable that reduces the matrix along the axis corresponding to ``out``.
    out : bool
        If ``True`` aggregate accessibility for origin rows, otherwise aggregate for destination columns.
    accessibility_column : str
        Name of the resulting column containing aggregated values.

    Returns
    -------
    pandas.DataFrame
        Single-column dataframe containing aggregated accessibility values.

    Raises
    ------
    ValueError
        If the provided matrix violates basic accessibility matrix validation rules.
    """

    validate_accessibility_matrix(accessibility_matrix, check_squared=False)
    df = pd.DataFrame(index=accessibility_matrix.index if out else accessibility_matrix.columns)
    df[accessibility_column] = agg_func(accessibility_matrix.values, axis=1 if out else 0)
    return df


def accessibility(
    accessibility_matrix: pd.DataFrame,
    agg_func: Callable,
    out: bool = True,
    accessibility_column: str = ACCESSIBILITY_COLUMN,
) -> pd.DataFrame:
    """Aggregate accessibility values with a user-provided reducer.

    Parameters
    ----------
    accessibility_matrix : pandas.DataFrame
        Accessibility weights between origin and destination blocks.
    agg_func : Callable
        Aggregation callable applied along axis 1 (origins) when ``out`` is ``True`` and along axis 0 otherwise.
    out : bool, default True
        Determines whether the aggregation is calculated for origins or destinations.
    accessibility_column : str, default ``ACCESSIBILITY_COLUMN``
        Column name assigned to the aggregated result.

    Returns
    -------
    pandas.DataFrame
        Aggregated accessibility values for each block.
    """

    return _accessibility(accessibility_matrix, out=out, agg_func=agg_func, accessibility_column=accessibility_column)


def median_accessibility(accessibility_matrix: pd.DataFrame, out: bool = True) -> pd.DataFrame:
    """Calculate the median accessibility score for each block.

    Parameters
    ----------
    accessibility_matrix : pandas.DataFrame
        Accessibility weights between origins and destinations.
    out : bool, default True
        If ``True``, compute median accessibility for origins; otherwise for destinations.

    Returns
    -------
    pandas.DataFrame
        Median accessibility values named ``median_accessibility``.
    """

    return _accessibility(
        accessibility_matrix, out=out, agg_func=np.median, accessibility_column=MEDIAN_ACCESSIBILITY_COLUMN
    )


def mean_accessibility(accessibility_matrix: pd.DataFrame, out: bool = True) -> pd.DataFrame:
    """Calculate the mean accessibility score for each block.

    Parameters
    ----------
    accessibility_matrix : pandas.DataFrame
        Accessibility weights between origins and destinations.
    out : bool, default True
        If ``True``, compute mean accessibility for origins; otherwise for destinations.

    Returns
    -------
    pandas.DataFrame
        Mean accessibility values named ``mean_accessibility``.
    """

    return _accessibility(
        accessibility_matrix, out=out, agg_func=np.mean, accessibility_column=MEAN_ACCESSIBILITY_COLUMN
    )


def max_accessibility(accessibility_matrix: pd.DataFrame, out: bool = True) -> pd.DataFrame:
    """Calculate the maximum accessibility value for each block.

    Parameters
    ----------
    accessibility_matrix : pandas.DataFrame
        Accessibility weights between origins and destinations.
    out : bool, default True
        If ``True``, compute the maximum for origins; otherwise for destinations.

    Returns
    -------
    pandas.DataFrame
        Maximum accessibility values named ``max_accessibility``.
    """

    return _accessibility(accessibility_matrix, out=out, agg_func=np.max, accessibility_column=MAX_ACCESSIBILITY_COLUMN)


__all__ = [
    "accessibility",
    "median_accessibility",
    "mean_accessibility",
    "max_accessibility",
]
