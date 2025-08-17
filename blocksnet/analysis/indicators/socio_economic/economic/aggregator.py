import pandas as pd
from .indicator import EconomicIndicator
from ..base_aggregator import BaseAggregator


class EconomicAggregator(BaseAggregator):

    indicator_cls = EconomicIndicator

    def __init__(
        self,
        parent_population_before: int,
        parent_population_after: int,
        child_population_before: int,
        child_population_after: int,
    ):
        super().__init__()
        self.parent_population_before = parent_population_before
        self.parent_population_after = parent_population_after
        self.child_population_before = child_population_before
        self.child_population_after = child_population_after

    def _aggregate(self, parent_before: pd.Series, child_before: pd.Series, child_after: pd.Series) -> pd.Series:
        parent_after = super()._aggregate(parent_before, child_before, child_after)
        for indicator in parent_after.index:
            if indicator in [
                EconomicIndicator.AVERAGE_WAGE,
                EconomicIndicator.FIXED_CAPITAL_INVESTMENT_PER_CAPITA,
                EconomicIndicator.GRP_PER_CAPITA,
            ]:
                ppb = self.parent_population_before
                ppa = self.parent_population_after
                cpb = self.child_population_before
                cpa = self.child_population_after
                pb = parent_before[indicator]
                cb = child_before[indicator]
                ca = child_after[indicator]
                parent_after[indicator] = (pb * ppb - cb * cpb + ca * cpa) / ppa
        return parent_after
