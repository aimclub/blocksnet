import pandas as pd
from functools import wraps
from pandera.typing import Series
from pandera import Field
from ...utils.validation import DfSchema, validate_accessibility_matrix

MEDIAN_ACCESSIBILITY_COLUMN = "median_accessibility"
MAX_ACCESSIBILITY_COLUMN = "max_accessibility"
AREA_ACCESSIBILITY_COLUMN = "area_accessibility"
RELATIVE_ACCESSIBILITY_COLUMN = "relative_accessibility"


def _validate_accessibility_matrix(func):
    """Accessibility matrix validation decorator"""

    @wraps(func)
    def wrapper(accessibility_matrix: pd.DataFrame, *args, **kwargs):
        validate_accessibility_matrix(accessibility_matrix)
        return func(accessibility_matrix, *args, **kwargs)

    return wrapper


@_validate_accessibility_matrix
def median_accessibility(accessibility_matrix: pd.DataFrame, out: bool = True) -> pd.DataFrame:
    df = pd.DataFrame(index=accessibility_matrix.index)
    df[MEDIAN_ACCESSIBILITY_COLUMN] = accessibility_matrix.median(axis=1 if out else 0)
    return df


@_validate_accessibility_matrix
def max_accessibility(accessibility_matrix: pd.DataFrame, out: bool = True) -> pd.DataFrame:
    df = pd.DataFrame(index=accessibility_matrix.index)
    df[MAX_ACCESSIBILITY_COLUMN] = accessibility_matrix.max(axis=1 if out else 0)
    return df


class AreaAccessibilityBlocksSchema(DfSchema):
    site_area: Series[float] = Field(ge=0)


@_validate_accessibility_matrix
def area_accessibility(accessibility_matrix: pd.DataFrame, blocks_df: pd.DataFrame):
    if not blocks_df.index.isin(accessibility_matrix.index).all():
        raise ValueError("Block index must be in matrix index")
    blocks_df = AreaAccessibilityBlocksSchema(blocks_df)
    weights = blocks_df.site_area
    result = (accessibility_matrix.mul(weights)).sum(axis=1) / weights.sum()
    return pd.DataFrame(result, index=blocks_df.index, columns=[AREA_ACCESSIBILITY_COLUMN])


@_validate_accessibility_matrix
def relative_accessibility(accessibility_matrix: pd.DataFrame, i: int, out: bool = True):
    if not i in accessibility_matrix.index:
        raise ValueError("i must be in matrix index")
    df = pd.DataFrame(index=accessibility_matrix.index)
    df[RELATIVE_ACCESSIBILITY_COLUMN] = accessibility_matrix.loc[i] if out else accessibility_matrix[i]
    return df
