import pandas as pd
from functools import wraps
from blocksnet.relations import validate_accessibility_matrix


def validate_accessibility_matrix(func):
    """Accessibility matrix validation decorator"""

    @wraps(func)
    def wrapper(accessibility_matrix: pd.DataFrame, *args, **kwargs):
        validate_accessibility_matrix(accessibility_matrix)
        return func(accessibility_matrix, *args, **kwargs)

    return wrapper
