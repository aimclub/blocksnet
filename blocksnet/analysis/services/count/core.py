import pandas as pd
from .schemas import BlocksServicesSchema

COUNT_COLUMN = "count"

COUNT_PREFIX = "count_"
BUILDINGS_SUFFIX = "_buildings"


def services_count(blocks_df: pd.DataFrame):
    """Summarise service counts and totals for each block.

    Parameters
    ----------
    blocks_df : pandas.DataFrame
        Dataframe containing service count columns prefixed with ``count_``.

    Returns
    -------
    pandas.DataFrame
        Dataframe with individual service counts and an aggregated ``count``
        column per block.

    Raises
    ------
    ValueError
        If no count columns are present or schema validation fails.
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
