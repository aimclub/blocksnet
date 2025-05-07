import pandas as pd
import numpy as np
from loguru import logger
from .schemas import BlocksServicesSchema
from ....utils.validation import ensure_crs

COUNT_COLUMN = "count"
SHANNON_DIVERSITY_COLUMN = "shannon_diversity"


def _shannon_index(series: pd.Series) -> float:
    series = series[series > 0]
    if len(series) == 0:
        return 0
    proportions = series / sum(series)
    return -np.sum(proportions * np.log(proportions)) + 0.0  # to exclude -0.0 result


def _concat_dfs(dfs: list[pd.DataFrame] | dict[str, pd.DataFrame]):

    for df in dfs:
        if not all(df.index == dfs[0].index):
            logger.warning("Index do not perfectly match. This might cause troubles and invalid results")

    dfs = [BlocksServicesSchema(df) for df in dfs]
    df = pd.concat(dfs, axis=1)
    df.columns = [f"count_{i}" for i, _ in enumerate(dfs)]

    return df.fillna(0)


def shannon_diversity(blocks_dfs: list[pd.DataFrame]):
    blocks_df = _concat_dfs(blocks_dfs)
    blocks_df[COUNT_COLUMN] = blocks_df.apply(sum, axis=1)
    blocks_df[SHANNON_DIVERSITY_COLUMN] = blocks_df.apply(_shannon_index, axis=1)
    return blocks_df[[COUNT_COLUMN, SHANNON_DIVERSITY_COLUMN]].copy()
