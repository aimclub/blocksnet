"""Area-weighted accessibility indicators."""

import pandas as pd
from blocksnet.relations import validate_accessibility_matrix

from .schemas import AreaAccessibilityBlocksSchema

AREA_ACCESSIBILITY_COLUMN = "area_accessibility"


def area_accessibility(
    accessibility_matrix: pd.DataFrame,
    blocks_df: pd.DataFrame,
    out: bool = True,
) -> pd.DataFrame:
    """Calculate an area-weighted accessibility score for blocks.

    Parameters
    ----------
    accessibility_matrix : pandas.DataFrame
        Accessibility weights between origin blocks (rows) and destination blocks (columns).
    blocks_df : pandas.DataFrame
        Block attributes containing ``site_area``. The dataframe is validated with
        :class:`AreaAccessibilityBlocksSchema` to ensure non-negative areas.
    out : bool, default True
        If ``True``, aggregate accessibility for origins (matrix rows). If ``False``, compute
        aggregated accessibility for destinations (matrix columns).

    Returns
    -------
    pandas.DataFrame
        A single-column dataframe named ``area_accessibility`` with an area-weighted accessibility
        value for each block.

    Raises
    ------
    ValueError
        If the matrix labels are inconsistent with the validated block index depending on the
        ``out`` orientation.
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
    if not accessibility_matrix.index.isin(blocks_df.index).all():
        raise ValueError("Accessibility matrix index must be in blocks index")

    s_rows = blocks_df.loc[accessibility_matrix.index, "site_area"]
    result = (accessibility_matrix.mul(s_rows, axis=0)).sum(axis=0) / s_rows.sum()
    return pd.DataFrame(result, index=accessibility_matrix.columns, columns=[AREA_ACCESSIBILITY_COLUMN])
