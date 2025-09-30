import pandas as pd
from .schemas import BlocksServicesSchema

COUNT_COLUMN = "count"

COUNT_PREFIX = "count_"
BUILDINGS_SUFFIX = "_buildings"


def services_count(blocks_df: pd.DataFrame):
    """Services count.

    Parameters
    ----------
    blocks_df : pd.DataFrame
        Description.

    """
    columns = [c for c in blocks_df.columns if COUNT_PREFIX in c and not BUILDINGS_SUFFIX in c]
    if len(columns) == 0:
        raise ValueError(
            f'Input DataFrame must have at least one "{COUNT_PREFIX}" including column, "{BUILDINGS_SUFFIX}" columns excluded'
        )
    dfs = {}
    for column in columns:
        df = blocks_df[[column]].rename(columns={column: COUNT_COLUMN})
        dfs[column] = BlocksServicesSchema(df)
    df = pd.concat(dfs.values(), axis=1)
    df.columns = list(dfs.keys())
    df[COUNT_COLUMN] = df.sum(axis=1)
    return df
