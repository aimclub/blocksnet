from enum import Enum
from abc import ABC, abstractmethod
import pandas as pd
from loguru import logger

CHILD_VALUE_BEFORE_COLUMN = "child_value_before"
CHILD_VALUE_AFTER_COLUMN = "child_value_after"
PARENT_VALUE_BEFORE_COLUMN = "parent_value_before"
PARENT_VALUE_AFTER_COLUMN = "parent_value_after"


class BaseAggregator(ABC):
    """Class to operate with child and parent territory indicators values in case of effects"""

    indicator_cls = Enum

    def __init__(self):
        self._indicators_data: dict[Enum, dict[str, float | int]] = {}

    @classmethod
    def _validate_indicator_type(cls, indicator):
        if not isinstance(indicator, cls.indicator_cls):
            expected_name = cls.indicator_cls.__name__
            actual_name = type(indicator).__name__
            raise ValueError(f"Indicator must be an instance of {expected_name}, but got {actual_name}")

    def add(
        self,
        indicator: Enum,
        parent_value_before: int | float,
        child_value_before: int | float,
        child_value_after: int | float,
    ):
        self._validate_indicator_type(indicator)
        self._indicators_data[indicator] = {
            CHILD_VALUE_BEFORE_COLUMN: child_value_before,
            CHILD_VALUE_AFTER_COLUMN: child_value_after,
            PARENT_VALUE_BEFORE_COLUMN: parent_value_before,
        }

    def delete(self, indicator: Enum):
        self._validate_indicator_type(indicator)
        del self._indicators_data[indicator]

    def _aggregate(self, parent_before: pd.Series, child_before: pd.Series, child_after: pd.Series) -> pd.Series:
        return parent_before - child_before + child_after

    def aggregate(self) -> pd.DataFrame:
        if len(self._indicators_data) == 0:
            raise ValueError("Cannot make DataFrame as no data was added")
        if len(self._indicators_data) < len(self.indicator_cls):
            logger.warning(f"Only {len(self._indicators_data)}/{len(self.indicator_cls)} indicators were added")
        df = pd.DataFrame.from_dict(self._indicators_data, orient="index")
        df[PARENT_VALUE_AFTER_COLUMN] = self._aggregate(
            df[PARENT_VALUE_BEFORE_COLUMN], df[CHILD_VALUE_BEFORE_COLUMN], df[CHILD_VALUE_AFTER_COLUMN]
        )
        return df
