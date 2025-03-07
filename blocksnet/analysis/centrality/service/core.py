import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from .schemas import BlocksSchema
from ...provision.diversity.core import shannon_diversity, COUNT_COLUMN, SHANNON_DIVERSITY_COLUMN
from ...network.accessibility.core import mean_accessibility, MEAN_ACCESSIBILITY_COLUMN
from ....common.validation import validate_matrix

CONNECTIVITY_COLUMN = "connectivity"
DENSITY_COLUMN = "density"
SERVICE_CENTRALITY_COLUMN = "service_centrality"


def service_centrality(
    accessibility_matrix: pd.DataFrame,
    blocks_df: pd.DataFrame,
    services_dfs: list[pd.DataFrame] = [],
    diversity_weight: float = 1,
    density_weight: float = 1,
    connectivity_weight: float = 1,
):
    validate_matrix(accessibility_matrix, blocks_df)
    blocks_df = BlocksSchema(blocks_df)

    accessibility_df = mean_accessibility(accessibility_matrix, out=False)[[MEAN_ACCESSIBILITY_COLUMN]]
    blocks_df = blocks_df.join(accessibility_df)
    blocks_df[CONNECTIVITY_COLUMN] = 1 / blocks_df[MEAN_ACCESSIBILITY_COLUMN]

    diversity_df = shannon_diversity(services_dfs)[[COUNT_COLUMN, SHANNON_DIVERSITY_COLUMN]]
    blocks_df = blocks_df.join(diversity_df)
    blocks_df[DENSITY_COLUMN] = blocks_df[COUNT_COLUMN] / blocks_df.site_area

    blocks_df = blocks_df.fillna(0)
    scaler = MinMaxScaler()
    normalized = scaler.fit_transform(blocks_df[[SHANNON_DIVERSITY_COLUMN, DENSITY_COLUMN, CONNECTIVITY_COLUMN]])

    sum_weight = diversity_weight + density_weight + connectivity_weight
    blocks_df[SERVICE_CENTRALITY_COLUMN] = [
        (diversity_weight * arr[0] + density_weight * arr[1] + connectivity_weight * arr[2]) / sum_weight
        for arr in normalized
    ]

    return blocks_df[[SHANNON_DIVERSITY_COLUMN, DENSITY_COLUMN, CONNECTIVITY_COLUMN, SERVICE_CENTRALITY_COLUMN]].copy()
