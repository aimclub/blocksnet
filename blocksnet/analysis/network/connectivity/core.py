import pandas as pd
from .schemas import BlocksAccessibilitySchema

CONNECTIVITY_COLUMN = "connectivity"
ACCESSIBILITY_SUFFIX = "_accessibility"


def _preprocess_and_validate(accessibility_df: pd.DataFrame) -> pd.DataFrame:
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
    """Convert accessibility values into connectivity scores.

    Parameters
    ----------
    accessibility_df : pandas.DataFrame
        Dataframe containing a single accessibility column identified by the
        ``"_accessibility"`` suffix.

    Returns
    -------
    pandas.DataFrame
        Dataframe with a ``connectivity`` column equal to the inverse of
        accessibility.

    Raises
    ------
    ValueError
        If no unique accessibility column can be identified or schema
        validation fails.
    """
    accessibility_df = _preprocess_and_validate(accessibility_df)
    accessibility_df[CONNECTIVITY_COLUMN] = 1.0 / accessibility_df["accessibility"]
    return accessibility_df[[CONNECTIVITY_COLUMN]].copy()
