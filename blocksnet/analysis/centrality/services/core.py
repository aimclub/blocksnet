import pandas as pd
from typing import Callable
from sklearn.preprocessing import MinMaxScaler
from blocksnet.analysis.network.accessibility import mean_accessibility

CONNECTIVITY_COLUMN = "connectivity"
DIVERSITY_COLUMN = "diversity"
DENSITY_COLUMN = "density"

SERVICES_CENTRALITY_COLUMN = "services_centrality"


def _calculate_connectivity(
    accessibility_matrix: pd.DataFrame, accessibility_func: Callable[[pd.DataFrame], pd.DataFrame]
) -> pd.Series:

    from blocksnet.analysis.network.connectivity.core import connectivity, CONNECTIVITY_COLUMN

    accessibility_df = accessibility_func(accessibility_matrix)
    connectivity_df = connectivity(accessibility_df)
    return connectivity_df[CONNECTIVITY_COLUMN]


def _calculate_diversity(blocks_df: pd.DataFrame) -> pd.Series:

    from blocksnet.analysis.diversity.shannon.core import shannon_diversity, SHANNON_DIVERSITY_COLUMN

    diversity_df = shannon_diversity(blocks_df)
    return diversity_df[SHANNON_DIVERSITY_COLUMN]


def _calculate_density(blocks_df: pd.DataFrame) -> pd.Series:

    from blocksnet.analysis.services.density.core import services_density, DENSITY_COLUMN

    density_df = services_density(blocks_df)
    return density_df[DENSITY_COLUMN]


def _preprocess_weights(weights: dict[str, float]) -> dict[str, float]:
    weights = weights.copy()
    keys = [CONNECTIVITY_COLUMN, DIVERSITY_COLUMN, DENSITY_COLUMN]
    for key, value in weights.items():
        if not key in keys:
            raise KeyError(f'{key} must be in {str.join(", ", keys)}')
        if not isinstance(value, (int, float)):
            raise ValueError(f"{key} weight must be float or int, got {type(value).__name__}")
        if value < 0:
            raise ValueError(f"{key} weight must not be <0, got {value}")
    for key in keys:
        if not key in weights:
            weights[key] = 1.0
    return weights


def services_centrality(
    accessibility_matrix: pd.DataFrame,
    blocks_df: pd.DataFrame,
    weights: dict[str, float] | None = None,
    accessibility_func: Callable[[pd.DataFrame], pd.DataFrame] = mean_accessibility,
) -> pd.DataFrame:
    weights = _preprocess_weights(weights or {})

    blocks_df = blocks_df.copy()
    columns = [CONNECTIVITY_COLUMN, DIVERSITY_COLUMN, DENSITY_COLUMN]
    blocks_df[CONNECTIVITY_COLUMN] = _calculate_connectivity(accessibility_matrix, accessibility_func)
    blocks_df[DIVERSITY_COLUMN] = _calculate_diversity(blocks_df)
    blocks_df[DENSITY_COLUMN] = _calculate_density(blocks_df)
    blocks_df = blocks_df[columns].fillna(0.0)

    scaler = MinMaxScaler()
    normalized = scaler.fit_transform(blocks_df[columns])

    sum_weight = sum(weights.values())

    blocks_df[SERVICES_CENTRALITY_COLUMN] = [
        sum(weights[col] * val for col, val in zip(columns, row)) / sum_weight for row in normalized
    ]

    return blocks_df[[*columns, SERVICES_CENTRALITY_COLUMN]].copy()
