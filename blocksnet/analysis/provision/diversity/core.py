import pandas as pd
import numpy as np
from loguru import logger
from .schemas import BlocksSchema

SHANNON_DIVERSITY_COLUMN = "shannon_diversity"


def _shannon_index(series: pd.Series) -> float:
    sum_count = sum(series)
    if sum_count == 0:
        return 0
    proportions = series / sum_count
    return -np.sum(proportions * np.log(proportions)) + 0.0  # to exclude -0.0 result


def _concat_dfs(dfs: list[pd.DataFrame] | dict[str, pd.DataFrame]):
    if isinstance(dfs, dict):
        columns = dfs.keys()
        dfs = list(dfs.values())
    elif isinstance(dfs, list):
        columns = [i for i, _ in enumerate(dfs)]

    for df in dfs:
        if not all(df.index == dfs[0].index):
            logger.warning("Index do not perfectly match. This might cause troubles and invalid results.")

    dfs = [BlocksSchema(df) for df in dfs]
    df = pd.concat(dfs, axis=1)
    df.columns = [f"count_{column}" for column in columns]

    return df.fillna(0)


def shannon_diversity(blocks_dfs: list[pd.DataFrame] | dict[str, pd.DataFrame]):
    blocks_df = _concat_dfs(blocks_dfs)
    blocks_df[SHANNON_DIVERSITY_COLUMN] = blocks_df.apply(_shannon_index, axis=1)
    return blocks_df
