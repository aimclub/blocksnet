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
    """Accessibility.

    Parameters
    ----------
    accessibility_matrix : pd.DataFrame
        Description.
    agg_func : Callable
        Description.
    out : bool
        Description.
    accessibility_column : str
        Description.

    Returns
    -------
    pd.DataFrame
        Description.

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
    """Accessibility.

    Parameters
    ----------
    accessibility_matrix : pd.DataFrame
        Description.
    agg_func : Callable
        Description.
    out : bool, default: True
        Description.
    accessibility_column : str, default: ACCESSIBILITY_COLUMN
        Description.

    Returns
    -------
    pd.DataFrame
        Description.

    """
    return _accessibility(accessibility_matrix, out=out, agg_func=agg_func, accessibility_column=accessibility_column)


def median_accessibility(accessibility_matrix: pd.DataFrame, out: bool = True) -> pd.DataFrame:
    """Median accessibility.

    Parameters
    ----------
    accessibility_matrix : pd.DataFrame
        Description.
    out : bool, default: True
        Description.

    Returns
    -------
    pd.DataFrame
        Description.

    """
    return _accessibility(
        accessibility_matrix, out=out, agg_func=np.median, accessibility_column=MEDIAN_ACCESSIBILITY_COLUMN
    )


def mean_accessibility(accessibility_matrix: pd.DataFrame, out: bool = True) -> pd.DataFrame:
    """Mean accessibility.

    Parameters
    ----------
    accessibility_matrix : pd.DataFrame
        Description.
    out : bool, default: True
        Description.

    Returns
    -------
    pd.DataFrame
        Description.

    """
    return _accessibility(
        accessibility_matrix, out=out, agg_func=np.mean, accessibility_column=MEAN_ACCESSIBILITY_COLUMN
    )


def max_accessibility(accessibility_matrix: pd.DataFrame, out: bool = True) -> pd.DataFrame:
    """Max accessibility.

    Parameters
    ----------
    accessibility_matrix : pd.DataFrame
        Description.
    out : bool, default: True
        Description.

    Returns
    -------
    pd.DataFrame
        Description.

    """
    return _accessibility(accessibility_matrix, out=out, agg_func=np.max, accessibility_column=MAX_ACCESSIBILITY_COLUMN)


__all__ = [
    "accessibility",
    "median_accessibility",
    "mean_accessibility",
    "max_accessibility",
]
