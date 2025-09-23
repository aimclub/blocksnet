import pandas as pd
from blocksnet.relations import validate_accessibility_matrix

RELATIVE_ACCESSIBILITY_COLUMN = "relative_accessibility"


def relative_accessibility(accessibility_matrix: pd.DataFrame, i: int, out: bool = True) -> pd.DataFrame:
    validate_accessibility_matrix(accessibility_matrix, check_squared=False)
    if not i in (accessibility_matrix.index if out else accessibility_matrix.columns):
        raise ValueError(f"i={i} must be in matrix {'index' if out else 'columns'}")
    df = pd.DataFrame(index=accessibility_matrix.index)
    df[RELATIVE_ACCESSIBILITY_COLUMN] = accessibility_matrix.loc[i] if out else accessibility_matrix[i]
    return df
