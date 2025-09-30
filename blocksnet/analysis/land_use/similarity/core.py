import pandas as pd
from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity
from blocksnet.config import service_types_config

LAND_USE_COLUMN = "land_use"
PROBABILITY_COLUMN = "probability"


def _get_blocks_df(blocks_df: pd.DataFrame, land_use_df: pd.DataFrame) -> pd.DataFrame:
    """Get blocks df.

    Parameters
    ----------
    blocks_df : pd.DataFrame
        Description.
    land_use_df : pd.DataFrame
        Description.

    Returns
    -------
    pd.DataFrame
        Description.

    """
    for column in land_use_df.columns:
        if column not in blocks_df.columns:
            blocks_df[column] = False

    return blocks_df[land_use_df.columns].fillna(0) > 0


def _get_land_use_df() -> pd.DataFrame:
    """Get land use df.

    Returns
    -------
    pd.DataFrame
        Description.

    """
    return service_types_config.land_use.T


def _calculate_cosine_similarity(blocks_df: pd.DataFrame, land_use_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate cosine similarity.

    Parameters
    ----------
    blocks_df : pd.DataFrame
        Description.
    land_use_df : pd.DataFrame
        Description.

    Returns
    -------
    pd.DataFrame
        Description.

    """
    similarity_mx = cosine_similarity(blocks_df.values, land_use_df.values)
    similarity_df = pd.DataFrame(similarity_mx, index=blocks_df.index, columns=land_use_df.index)
    return similarity_df


def _preprocess_and_validate(blocks_df: pd.DataFrame) -> pd.DataFrame:

    """Preprocess and validate.

    Parameters
    ----------
    blocks_df : pd.DataFrame
        Description.

    Returns
    -------
    pd.DataFrame
        Description.

    """
    from blocksnet.analysis.services.count.core import services_count, COUNT_COLUMN, COUNT_PREFIX

    logger.info("Preprocessing and validating input data")
    blocks_df = services_count(blocks_df).drop(columns=[COUNT_COLUMN])
    columns = {}
    for column in blocks_df.columns:
        service_type = column.removeprefix(COUNT_PREFIX)
        if service_type in service_types_config:
            columns[column] = service_type
        else:
            logger.warning(f'Unknown service type "{service_type}" will be ignored')
    logger.info(f"{len(columns)}/{len(service_types_config.service_types)} service types are provided")
    return blocks_df.rename(columns=columns)


def land_use_similarity(
    blocks_df: pd.DataFrame,
) -> pd.DataFrame:
    """Land use similarity.

    Parameters
    ----------
    blocks_df : pd.DataFrame
        Description.

    Returns
    -------
    pd.DataFrame
        Description.

    """
    blocks_df = _preprocess_and_validate(blocks_df)
    land_use_df = _get_land_use_df()
    blocks_df = _get_blocks_df(blocks_df, land_use_df)

    similarity_df = _calculate_cosine_similarity(blocks_df, land_use_df)
    land_use_series = similarity_df.idxmax(axis=1)
    probability_series = similarity_df.max(axis=1)
    similarity_df[LAND_USE_COLUMN] = land_use_series
    similarity_df[PROBABILITY_COLUMN] = probability_series
    similarity_df.loc[similarity_df.probability == 0, "land_use"] = None

    return similarity_df
