import pandas as pd
import numpy as np
from blocksnet.config import log_config
from loguru import logger
from blocksnet.analysis.services.count.core import services_count, COUNT_PREFIX, COUNT_COLUMN

SHANNON_DIVERSITY_COLUMN = "shannon_diversity"


def _shannon_index(series: pd.Series) -> float:
    count = series[COUNT_COLUMN]
    index = [i for i in series.index if COUNT_PREFIX in i]
    series = series[index]
    series = series[series > 0]
    if len(series) == 0:
        return 0
    proportions = series / count
    return -np.sum(proportions * np.log(proportions)) + 0.0  # to exclude -0.0 result


def shannon_diversity(blocks_df: pd.DataFrame):
    count_df = services_count(blocks_df)
    logger.info("Calculating Shannon diversity index")
    if log_config.disable_tqdm:
        diversity = count_df.apply(_shannon_index, axis=1)
    else:
        diversity = count_df.progress_apply(_shannon_index, axis=1)
    count_df[SHANNON_DIVERSITY_COLUMN] = diversity
    return count_df
