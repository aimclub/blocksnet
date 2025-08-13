# Импорты библиотек
import pandas as pd
from .schemas import BlocksSchema

CO_OCCURRENCE_COLUMN = "co_occurrence"
NORMALIZED_COLUMN = "normalized_value"


def _intersection(df1, df2) -> int:
    return len(df1[(df1["count"] > 0) & (df2["count"] > 0)])


def _intersection_matrix(services_dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    mx = pd.DataFrame(0, index=services_dfs.keys(), columns=services_dfs.keys())
    for i in mx.index:
        for j in mx.columns:
            mx.loc[i, j] = _intersection(services_dfs[i], services_dfs[j])
    return mx


def _union(df1, df2) -> int:
    return len(df1[(df1["count"] > 0) | (df2["count"] > 0)])


def _union_matrix(services_dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    mx = pd.DataFrame(0, index=services_dfs.keys(), columns=services_dfs.keys())
    for i in mx.index:
        for j in mx.columns:
            mx.loc[i, j] = _union(services_dfs[i], services_dfs[j])
    return mx


def collocation_matrix(services_dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    services_dfs = {str(st): BlocksSchema(df) for st, df in services_dfs.items()}
    intersection_mx = _intersection_matrix(services_dfs)
    union_mx = _union_matrix(services_dfs)
    iou_mx = intersection_mx / union_mx

    for i in iou_mx.index:
        services_df = services_dfs[i]
        intersection = len(services_df[services_df["count"] > 1])
        union = len(services_df[services_df["count"] > 0])
        iou_mx.loc[i, i] = intersection / union

    return iou_mx
