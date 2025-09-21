import pandas as pd
from blocksnet.relations import validate_accessibility_matrix
from .schemas import AreaAccessibilityBlocksSchema

AREA_ACCESSIBILITY_COLUMN = "area_accessibility"


def area_accessibility(accessibility_matrix: pd.DataFrame, blocks_df: pd.DataFrame, out: bool = True):
    validate_accessibility_matrix(
        accessibility_matrix, blocks_df=blocks_df, index=out, columns=not out, check_squared=False
    )
    blocks_df = AreaAccessibilityBlocksSchema(blocks_df)
    weights = blocks_df.site_area
    result = (accessibility_matrix.mul(weights)).sum(axis=1 if out else 0) / weights.sum()
    return pd.DataFrame(result, index=blocks_df.index, columns=[AREA_ACCESSIBILITY_COLUMN])
