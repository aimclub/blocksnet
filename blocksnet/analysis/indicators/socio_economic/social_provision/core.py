import pandas as pd
from loguru import logger
from .indicator import SocialProvisionIndicator
from .schemas import BlocksSchema
from blocksnet.config import service_types_config


def calculate_social_provision_indicators(
    blocks_df: pd.DataFrame,
) -> tuple[dict[SocialProvisionIndicator, float], list[SocialProvisionIndicator]]:

    result = {}
    missing = []
    for indicator in SocialProvisionIndicator:

        name = indicator.meta.name
        if not name in service_types_config:
            logger.warning(f"{name} not found in config. The indicator is skipped")
            missing.append(indicator)
            continue

        _, demand, accessibility = service_types_config[name].values()

        column = f"capacity_{indicator.meta.name}"
        if not column in blocks_df.columns:
            logger.warning(f"{column} is missing. The indicator is skipped")
            missing.append(indicator)
            continue

        df = BlocksSchema(blocks_df.rename(columns={column: "capacity"}))

        population = df["population"].sum()
        capacity = df["capacity"].sum()
        provision = (population * demand / 1000) / capacity

        result[indicator] = min(1.0, float(provision))

    return result, missing
