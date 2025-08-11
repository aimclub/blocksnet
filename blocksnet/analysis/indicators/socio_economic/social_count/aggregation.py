import pandas as pd
from .indicator import SocialCountIndicator
from ..base_aggregation import BaseAggregation


class SocialCountAggregation(BaseAggregation):

    INDICATOR_CLS = SocialCountIndicator

    def _aggregate(self, parent_before: pd.Series, child_before: pd.Series, child_after: pd.Series) -> pd.Series:
        return parent_before - child_before + child_after
