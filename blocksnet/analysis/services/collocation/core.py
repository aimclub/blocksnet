# Импорты библиотек
import pandas as pd

CO_OCCURRENCE_COLUMN = "co_occurrence"
NORMALIZED_COLUMN = "normalized_value"


def _intersection(df1, df2) -> int:
    return len(df1[(df1 > 0) & (df2 > 0)])


def _intersection_matrix(blocks_df: pd.DataFrame) -> pd.DataFrame:
    mx = pd.DataFrame(0, index=blocks_df.columns, columns=blocks_df.columns)
    for st_a in mx.index:
        for st_b in mx.columns:
            mx.loc[st_a, st_b] = _intersection(blocks_df[st_a], blocks_df[st_b])
    return mx


def _union(df1, df2) -> int:
    return len(df1[(df1 > 0) | (df2 > 0)])


def _union_matrix(blocks_df: pd.DataFrame) -> pd.DataFrame:
    mx = pd.DataFrame(0, index=blocks_df.columns, columns=blocks_df.columns)
    for st_a in mx.index:
        for st_b in mx.columns:
            mx.loc[st_a, st_b] = _union(blocks_df[st_a], blocks_df[st_b])
    return mx


def _preprocess_and_validate(blocks_df: pd.DataFrame) -> pd.DataFrame:

    from ..count.core import services_count, COUNT_COLUMN, COUNT_PREFIX

    counts_df = services_count(blocks_df).drop(columns=[COUNT_COLUMN])
    columns = {c: c.removeprefix(COUNT_PREFIX) for c in counts_df.columns}
    return counts_df.rename(columns=columns)


def services_collocation(blocks_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate intersection-over-union of service presence pairs.

    Parameters
    ----------
    blocks_df : pandas.DataFrame
        Block dataframe containing service count columns prefixed with
        ``count_``.

    Returns
    -------
    pandas.DataFrame
        Symmetric matrix where each entry is the ratio of shared presence to
        total presence for a pair of services.

    Raises
    ------
    ValueError
        If service counts are missing or fail validation.
    """
    blocks_df = _preprocess_and_validate(blocks_df)
    intersection_mx = _intersection_matrix(blocks_df)
    union_mx = _union_matrix(blocks_df)
    iou_mx = intersection_mx / union_mx

    for i in iou_mx.index:
        series = blocks_df[i]
        intersection = len(series[series > 1])
        union = len(series[series > 0])
        iou_mx.loc[i, i] = intersection / union

    return iou_mx
