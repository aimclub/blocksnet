import pandas as pd
from .indicator import EngineeringIndicator
from blocksnet.analysis.services import services_count

SKIP_INDICATORS = [
    EngineeringIndicator.NON_GASIFIED_SETTLEMENTS,
    EngineeringIndicator.INFRASTRUCTURE_OBJECT,
]


def calculate_engineering_indicators(blocks_df: pd.DataFrame) -> dict[EngineeringIndicator, float]:

    blocks_df = services_count(blocks_df)

    count_indicators = [ind for ind in EngineeringIndicator if ind not in SKIP_INDICATORS]
    count_columns = [f"count_{ind.meta.name}" for ind in count_indicators]
    missing_columns = set(set(count_columns)).difference(blocks_df.columns)

    if len(missing_columns) > 0:
        missing_str = str.join(", ", missing_columns)
        raise RuntimeError(f"Missing columns: {missing_str}")

    result = {}
    for indicator in count_indicators:
        column = f"count_{indicator.meta.name}"
        count = blocks_df[column].sum()
        result[indicator] = int(count)

    count = sum(result.values())
    result[EngineeringIndicator.INFRASTRUCTURE_OBJECT] = count

    return result
