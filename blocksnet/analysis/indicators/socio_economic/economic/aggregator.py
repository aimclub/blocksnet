import pandas as pd
from .indicator import EconomicIndicator
from ..base_aggregator import BaseAggregator


class EconomicAggregator(BaseAggregator):

    indicator_cls = EconomicIndicator

    def __init__(self, parent_population_before: int, child_population_before: int, child_population_after: int):
        super().__init__()
        self.parent_population_before = parent_population_before
        self.child_population_before = child_population_before
        self.child_population_after = child_population_after

    @property
    def parent_population_after(self):
        return self.parent_population_before - self.child_population_before + self.child_population_after

    def _aggregate(self, parent_before: pd.Series, child_before: pd.Series, child_after: pd.Series) -> pd.Series:
        indicator_cls = self.indicator_cls
        parent_after = super()._aggregate(parent_before, child_before, child_after)
        for indicator in parent_after.index:
            if indicator in [
                indicator_cls.AVERAGE_WAGE,
                indicator_cls.FIXED_CAPITAL_INVESTMENT_PER_CAPITA,
                indicator_cls.GRP_PER_CAPITA,
            ]:
                ppb = self.parent_population_before
                ppa = self.parent_population_after
                cpb = self.child_population_before
                cpa = self.child_population_after
                pb = parent_before.loc[indicator]
                cb = child_before.loc[indicator]
                ca = child_after.loc[indicator]
                parent_after.loc[indicator] = (pb * ppb - cb * cpb + ca * cpa) / ppa
        return parent_after
