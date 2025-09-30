import pandas as pd
from .schemas import BlocksAccessibilitySchema

CONNECTIVITY_COLUMN = "connectivity"
ACCESSIBILITY_SUFFIX = "_accessibility"


def _preprocess_and_validate(accessibility_df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess and validate.

    Parameters
    ----------
    accessibility_df : pd.DataFrame
        Description.

    Returns
    -------
    pd.DataFrame
        Description.

    """
    columns = [c for c in accessibility_df.columns if ACCESSIBILITY_SUFFIX in c]
    if len(columns) > 1:
        raise ValueError(
            f'More than 1 columns were found with "{ACCESSIBILITY_SUFFIX}" in name: {str.join(", ", columns)}'
        )
    if len(columns) == 0:
        raise ValueError(f'No columns with "{ACCESSIBILITY_SUFFIX}" in name were found')
    column = columns[0]
    accessibility_df = accessibility_df.rename(columns={column: "accessibility"})
    return BlocksAccessibilitySchema(accessibility_df)


def calculate_connectivity(accessibility_df: pd.DataFrame):
    """Calculate connectivity.

    Parameters
    ----------
    accessibility_df : pd.DataFrame
        Description.

    """
    accessibility_df = _preprocess_and_validate(accessibility_df)
    accessibility_df[CONNECTIVITY_COLUMN] = 1.0 / accessibility_df["accessibility"]
    return accessibility_df[[CONNECTIVITY_COLUMN]].copy()
