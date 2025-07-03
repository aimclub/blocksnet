import pandas as pd
from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity
from blocksnet.config import service_types_config
from blocksnet.enums import LandUse
from .schemas import ServicesSchema

LAND_USE_COLUMN = "land_use"
PROBABILITY_COLUMN = "probability"


def _preprocess_input(services_dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    logger.info("Validating input data")

    if len(services_dfs) == 0:
        raise ValueError("services_dfs len must be greater than 0")

    for service_type in services_dfs.keys():
        if service_type not in service_types_config:
            raise KeyError(f"{service_type} is not known by service_types_config")

    logger.info(f"{len(services_dfs)}/{len(service_types_config.service_types)} service types are provided")
    services_dfs = {st: ServicesSchema(df) for st, df in services_dfs.items()}
    return services_dfs


def _get_blocks_df(services_dfs: dict[str, pd.DataFrame], land_use_df: pd.DataFrame) -> pd.DataFrame:
    dfs = [df.rename(columns={"count": st}) for st, df in services_dfs.items()]
    blocks_df = pd.concat(dfs, axis=1)

    for column in land_use_df.columns:
        if column not in blocks_df.columns:
            blocks_df[column] = False

    return blocks_df[land_use_df.columns].fillna(0) > 0


def _get_land_use_df() -> pd.DataFrame:
    return service_types_config.land_use.T


def _calculate_cosine_similarity(blocks_df: pd.DataFrame, land_use_df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Calculating cosine similarity")
    similarity_mx = cosine_similarity(blocks_df.values, land_use_df.values)
    similarity_df = pd.DataFrame(similarity_mx, index=blocks_df.index, columns=land_use_df.index)
    return similarity_df


def land_use_similarity(
    services_dfs: dict[str, pd.DataFrame],
):
    services_dfs = _preprocess_input(services_dfs)
    land_use_df = _get_land_use_df()
    blocks_df = _get_blocks_df(services_dfs, land_use_df)

    similarity_df = _calculate_cosine_similarity(blocks_df, land_use_df)
    land_use_series = similarity_df.idxmax(axis=1)
    probability_series = similarity_df.max(axis=1)
    similarity_df[LAND_USE_COLUMN] = land_use_series
    similarity_df[PROBABILITY_COLUMN] = probability_series
    similarity_df.loc[similarity_df.probability == 0, "land_use"] = None

    return similarity_df
