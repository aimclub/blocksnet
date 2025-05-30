import numpy as np
import pandas as pd

from .schemas import AreaAccessibilityBlocksSchema
from .utils import validate_accessibility_matrix


MEDIAN_ACCESSIBILITY_COLUMN = "median_accessibility"
MEAN_ACCESSIBILITY_COLUMN = "mean_accessibility"
MAX_ACCESSIBILITY_COLUMN = "max_accessibility"
AREA_ACCESSIBILITY_COLUMN = "area_accessibility"
RELATIVE_ACCESSIBILITY_COLUMN = "relative_accessibility"


@validate_accessibility_matrix
def median_accessibility(accessibility_matrix: pd.DataFrame, out: bool = True) -> pd.DataFrame:
    df = pd.DataFrame(index=accessibility_matrix.index)
    df[MEDIAN_ACCESSIBILITY_COLUMN] = np.median(accessibility_matrix.values, axis=1 if out else 0)
    return df


@validate_accessibility_matrix
def mean_accessibility(accessibility_matrix: pd.DataFrame, out: bool = True) -> pd.DataFrame:
    df = pd.DataFrame(index=accessibility_matrix.index)
    df[MEAN_ACCESSIBILITY_COLUMN] = np.mean(accessibility_matrix.values, axis=1 if out else 0)
    return df


@validate_accessibility_matrix
def max_accessibility(accessibility_matrix: pd.DataFrame, out: bool = True) -> pd.DataFrame:
    df = pd.DataFrame(index=accessibility_matrix.index)
    df[MAX_ACCESSIBILITY_COLUMN] = np.max(accessibility_matrix.values, axis=1 if out else 0)
    return df


@validate_accessibility_matrix
def area_accessibility(accessibility_matrix: pd.DataFrame, blocks_df: pd.DataFrame):
    blocks_df = AreaAccessibilityBlocksSchema(blocks_df)
    weights = blocks_df.site_area
    result = (accessibility_matrix.mul(weights)).sum(axis=1) / weights.sum()
    return pd.DataFrame(result, index=blocks_df.index, columns=[AREA_ACCESSIBILITY_COLUMN])


@validate_accessibility_matrix
def relative_accessibility(accessibility_matrix: pd.DataFrame, i: int, out: bool = True):
    if not i in accessibility_matrix.index:
        raise ValueError("i must be in matrix index")
    df = pd.DataFrame(index=accessibility_matrix.index)
    df[RELATIVE_ACCESSIBILITY_COLUMN] = accessibility_matrix.loc[i] if out else accessibility_matrix[i]
    return df


__all__ = [
    "median_accessibility",
    "mean_accessibility",
    "max_accessibility",
    "area_accessibility",
    "relative_accessibility",
]
