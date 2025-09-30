import pandas as pd
from blocksnet.relations import validate_accessibility_matrix
from .schemas import AreaAccessibilityBlocksSchema

AREA_ACCESSIBILITY_COLUMN = "area_accessibility"


def area_accessibility(
    accessibility_matrix: pd.DataFrame,
    blocks_df: pd.DataFrame,
    out: bool = True,
) -> pd.DataFrame:

    """Area accessibility.

    Parameters
    ----------
    accessibility_matrix : pd.DataFrame
        Description.
    blocks_df : pd.DataFrame
        Description.
    out : bool, default: True
        Description.

    Returns
    -------
    pd.DataFrame
        Description.

    """
    validate_accessibility_matrix(
        accessibility_matrix,
        check_squared=False,
    )

    blocks_df = AreaAccessibilityBlocksSchema(blocks_df)

    if out:
        if not accessibility_matrix.columns.isin(blocks_df.index).all():
            raise ValueError("Accessibility matrix columns must be in blocks index")

        s_cols = blocks_df.loc[accessibility_matrix.columns, "site_area"]
        result = (accessibility_matrix.mul(s_cols, axis=1)).sum(axis=1) / s_cols.sum()
        return pd.DataFrame(result, index=accessibility_matrix.index, columns=[AREA_ACCESSIBILITY_COLUMN])
    else:
        if not accessibility_matrix.index.isin(blocks_df.index).all():
            raise ValueError("Accessibility matrix index must be in blocks index")

        s_rows = blocks_df.loc[accessibility_matrix.index, "site_area"]
        result = (accessibility_matrix.mul(s_rows, axis=0)).sum(axis=0) / s_rows.sum()
        return pd.DataFrame(result, index=accessibility_matrix.columns, columns=[AREA_ACCESSIBILITY_COLUMN])
