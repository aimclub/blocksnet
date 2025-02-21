import pandas as pd
from functools import wraps
from ....utils import validation


def validate_accessibility_matrix(func):
    """Accessibility matrix validation decorator"""

    @wraps(func)
    def wrapper(accessibility_matrix: pd.DataFrame, *args, **kwargs):
        validation.validate_accessibility_matrix(accessibility_matrix)
        return func(accessibility_matrix, *args, **kwargs)

    return wrapper
