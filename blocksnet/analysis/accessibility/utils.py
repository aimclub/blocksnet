import pandas as pd
from functools import wraps
from ...utils import validation


def validate_accessibility_matrix(func):
    """Accessibility matrix validation decorator"""

    @wraps(func)
    def wrapper(accessibility_matrix: pd.DataFrame, *args, **kwargs):
        validation.validate_matrix(accessibility_matrix)
        return func(accessibility_matrix, *args, **kwargs)

    return wrapper


@validate_accessibility_matrix
def get_context(
    accessibility_matrix: pd.DataFrame, blocks_ids: list[int], accessibility: float, out: bool = True, keep: bool = True
):
    for i in blocks_ids:
        if not i in accessibility_matrix.index:
            raise ValueError("blocks_ids must be in matrix index")
    if not out:
        accessibility_matrix = accessibility_matrix.transpose()
    acc_mx = accessibility_matrix[blocks_ids]
    if not keep:
        acc_mx = acc_mx[~acc_mx.index.isin(blocks_ids)]
    mask = (acc_mx <= accessibility).any(axis=1)
    return list(acc_mx[mask].index)
