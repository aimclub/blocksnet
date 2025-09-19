import pandas as pd
from .schemas import BlocksServicesSchema, BlocksAreaSchema

COUNT_COLUMN = "count"
DENSITY_COLUMN = "density"

COUNT_PREFIX = "count_"
BUILDINGS_SUFFIX = "_buildings"


def _preprocess_area(blocks_df: pd.DataFrame) -> pd.DataFrame:
    return BlocksAreaSchema(blocks_df)


def _calculate_counts(blocks_df: pd.DataFrame) -> pd.DataFrame:
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


def _preprocess_and_validate(blocks_df: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([area_df, counts_df], axis=1)


def services_density(blocks_df: pd.DataFrame):
    area_df = _preprocess_area(blocks_df)
    counts_df = _calculate_counts(blocks_df)
    blocks_df = pd.concat([area_df, counts_df], axis=1)
    blocks_df[DENSITY_COLUMN] = blocks_df[COUNT_COLUMN] / blocks_df.site_area
    return blocks_df
