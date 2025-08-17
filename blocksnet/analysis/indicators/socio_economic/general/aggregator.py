from enum import Enum
import pandas as pd
from .indicator import GeneralIndicator
from ..base_aggregator import BaseAggregator


class GeneralAggregator(BaseAggregator):

    indicator_cls = GeneralIndicator

    def _aggregate_urbanization(
        self, parent_before: pd.Series, parent_after: pd.Series, child_before: pd.Series, child_after: pd.Series
    ) -> float:
        if GeneralIndicator.AREA not in parent_after.index:
            raise KeyError("Area is required to aggregate urbanization")
        pab = parent_before[GeneralIndicator.AREA]
        pub = parent_before[GeneralIndicator.URBANIZATION]
        cab = child_before[GeneralIndicator.AREA]
        cub = child_before[GeneralIndicator.URBANIZATION]
        caa = child_after[GeneralIndicator.AREA]
        cua = child_after[GeneralIndicator.URBANIZATION]
        paa = parent_after.loc[GeneralIndicator.AREA]
        return (pab * pub - cab * cub + caa * cua) / paa

    def _aggregate(self, parent_before: pd.Series, child_before: pd.Series, child_after: pd.Series) -> pd.Series:
        parent_after = super()._aggregate(parent_before, child_before, child_after)
        for indicator in parent_after.index:
            if indicator == GeneralIndicator.URBANIZATION:
                parent_after[GeneralIndicator.URBANIZATION] = self._aggregate_urbanization(
                    parent_before, parent_after, child_before, child_after
                )
        return parent_after
