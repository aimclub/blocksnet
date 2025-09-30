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
    """Aggregate accessibility scores across the matrix axis.

    Parameters
    ----------
    accessibility_matrix : pandas.DataFrame
        Accessibility matrix where rows correspond to origins and columns to
        destinations.
    agg_func : Callable
        Reduction applied across the selected axis (``numpy`` aggregator or
        similar callable accepting ``axis`` keyword).
    out : bool, optional
        If ``True``, aggregate across destinations for each origin. If
        ``False``, aggregate across origins for each destination. Default is
        ``True``.
    accessibility_column : str, optional
        Name of the resulting column in the returned dataframe. Defaults to
        ``"accessibility"``.

    Returns
    -------
    pandas.DataFrame
        Dataframe indexed by origins (or destinations when ``out=False``) with
        a single accessibility column.

    Raises
    ------
    ValueError
        If ``accessibility_matrix`` fails validation against
        :func:`blocksnet.relations.validate_accessibility_matrix`.
    """
    return _accessibility(accessibility_matrix, out=out, agg_func=agg_func, accessibility_column=accessibility_column)


def median_accessibility(accessibility_matrix: pd.DataFrame, out: bool = True) -> pd.DataFrame:
    """Compute the median accessibility for each origin or destination.

    Parameters
    ----------
    accessibility_matrix : pandas.DataFrame
        Matrix of accessibility scores.
    out : bool, optional
        Whether to aggregate across destinations (``True``) or origins
        (``False``). Default is ``True``.

    Returns
    -------
    pandas.DataFrame
        Dataframe with a ``median_accessibility`` column.

    Raises
    ------
    ValueError
        If the matrix fails validation.
    """
    return _accessibility(
        accessibility_matrix, out=out, agg_func=np.median, accessibility_column=MEDIAN_ACCESSIBILITY_COLUMN
    )


def mean_accessibility(accessibility_matrix: pd.DataFrame, out: bool = True) -> pd.DataFrame:
    """Compute the mean accessibility for each origin or destination.

    Parameters
    ----------
    accessibility_matrix : pandas.DataFrame
        Matrix of accessibility scores.
    out : bool, optional
        Whether to aggregate across destinations (``True``) or origins
        (``False``). Default is ``True``.

    Returns
    -------
    pandas.DataFrame
        Dataframe with a ``mean_accessibility`` column.

    Raises
    ------
    ValueError
        If the matrix fails validation.
    """
    return _accessibility(
        accessibility_matrix, out=out, agg_func=np.mean, accessibility_column=MEAN_ACCESSIBILITY_COLUMN
    )


def max_accessibility(accessibility_matrix: pd.DataFrame, out: bool = True) -> pd.DataFrame:
    """Compute the maximum accessibility for each origin or destination.

    Parameters
    ----------
    accessibility_matrix : pandas.DataFrame
        Matrix of accessibility scores.
    out : bool, optional
        Whether to aggregate across destinations (``True``) or origins
        (``False``). Default is ``True``.

    Returns
    -------
    pandas.DataFrame
        Dataframe with a ``max_accessibility`` column.

    Raises
    ------
    ValueError
        If the matrix fails validation.
    """
    return _accessibility(accessibility_matrix, out=out, agg_func=np.max, accessibility_column=MAX_ACCESSIBILITY_COLUMN)


__all__ = [
    "accessibility",
    "median_accessibility",
    "mean_accessibility",
    "max_accessibility",
]
