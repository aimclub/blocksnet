# import pandas as pd
# from functools import wraps
# from blocksnet.relations import validate_accessibility_matrix as validate_acc_mx


# def validate_accessibility_matrix(func):
#     """Accessibility matrix validation decorator"""

#     @wraps(func)
#     def wrapper(accessibility_matrix: pd.DataFrame, *args, **kwargs):
#         validate_acc_mx(accessibility_matrix)
#         return func(accessibility_matrix, *args, **kwargs)

#     return wrapper
