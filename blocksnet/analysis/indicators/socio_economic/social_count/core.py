import pandas as pd
from loguru import logger
from .indicator import SocialCountIndicator
from blocksnet.analysis.services import services_count


def calculate_social_count_indicators(
    blocks_df: pd.DataFrame,
) -> tuple[dict[SocialCountIndicator, int], list[SocialCountIndicator]]:
    """Count social infrastructure units available within blocks.

    Parameters
    ----------
    blocks_df : pandas.DataFrame
        Block dataframe with service counts.

    Returns
    -------
    tuple[dict[SocialCountIndicator, int], list[SocialCountIndicator]]
        Tuple of indicator totals and a list of indicators missing data.
    """
    blocks_df = services_count(blocks_df)

    result = {}
    missing = []
    for indicator in SocialCountIndicator:
        column = f"count_{indicator.meta.name}"
        if column in blocks_df.columns:
            count = blocks_df[column].sum()
            result[indicator] = int(count)
        else:
            logger.warning(f"{column} is missing. The indicator is skipped")
            missing.append(indicator)

    return result, missing
