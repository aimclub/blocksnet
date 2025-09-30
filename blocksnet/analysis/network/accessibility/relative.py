"""Relative accessibility indicator utilities."""

import pandas as pd

from blocksnet.relations import validate_accessibility_matrix

RELATIVE_ACCESSIBILITY_COLUMN = "relative_accessibility"


def relative_accessibility(accessibility_matrix: pd.DataFrame, i: int, out: bool = True) -> pd.DataFrame:
    """Extract relative accessibility scores for a selected block.

    Parameters
    ----------
    accessibility_matrix : pandas.DataFrame
        Accessibility weights between origin and destination blocks.
    i : int
        Identifier of the block for which relative accessibility is requested. The identifier must
        exist either in the matrix index (for ``out=True``) or columns (for ``out=False``).
    out : bool, default True
        Orientation of the returned scores. ``True`` returns accessibility from block ``i`` to all
        destinations, while ``False`` returns accessibility to block ``i`` from all origins.

    Returns
    -------
    pandas.DataFrame
        A dataframe with a ``relative_accessibility`` column indexed by the origin blocks.

    Raises
    ------
    ValueError
        If ``i`` is absent from the relevant axis of the accessibility matrix.
    """

    validate_accessibility_matrix(accessibility_matrix, check_squared=False)
    if i not in (accessibility_matrix.index if out else accessibility_matrix.columns):
        raise ValueError(f"i={i} must be in matrix {'index' if out else 'columns'}")
    df = pd.DataFrame(index=accessibility_matrix.index)
    df[RELATIVE_ACCESSIBILITY_COLUMN] = accessibility_matrix.loc[i] if out else accessibility_matrix[i]
    return df
