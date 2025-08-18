from enum import Enum
from functools import wraps
from .indicator import IndicatorEnum
from .indicator.enums import GeneralIndicator, DemographicIndicator

CHILD_VALUE_BEFORE_COLUMN = "child_value_before"
CHILD_VALUE_AFTER_COLUMN = "child_value_after"
PARENT_VALUE_BEFORE_COLUMN = "parent_value_before"
PARENT_VALUE_AFTER_COLUMN = "parent_value_after"


def _validate_indicator_type(indicator: Enum):
    if not isinstance(indicator, IndicatorEnum):
        expected_name = IndicatorEnum.__name__
        actual_name = type(indicator).__name__
        raise TypeError(f"Indicator must be an instance of {expected_name}. Got {actual_name}")
    pass


def validate_indicator(require: bool = False):
    def deco(func):
        @wraps(func)
        def wrapper(self, indicator: IndicatorEnum, *args, **kwargs):
            _validate_indicator_type(indicator)
            if require and indicator not in self._data:
                cls = indicator.__class__.__name__
                name = getattr(indicator, "name", str(indicator))
                raise KeyError(f"Indicator {cls}.{name} not found in self.")
            return func(self, indicator, *args, **kwargs)

        return wrapper

    return deco


class SocioEconomicAggregator:
    def __init__(self):
        self._data: dict[IndicatorEnum, dict[str, float | int]] = {}

    @validate_indicator()
    def add(
        self,
        indicator: IndicatorEnum,
        child_value_before: int | float,
        child_value_after: int | float,
        parent_value_before: int | float,
        parent_value_after: int | float | None = None,
    ):
        data = {
            CHILD_VALUE_BEFORE_COLUMN: child_value_before,
            CHILD_VALUE_AFTER_COLUMN: child_value_after,
            PARENT_VALUE_BEFORE_COLUMN: parent_value_before,
        }
        if parent_value_after is not None:
            data[PARENT_VALUE_AFTER_COLUMN] = parent_value_after
        self._data[indicator] = data

    @validate_indicator(require=True)
    def delete(self, indicator: IndicatorEnum):
        del self._data[indicator]

    @validate_indicator(require=True)
    def get(self, indicator: IndicatorEnum):
        return self._data[indicator].copy()

    def _aggregate_normalized(self, data: dict[str, float | int], indicator_norm: IndicatorEnum) -> int | float:
        """In case of per population or per area"""
        cb_a = data[CHILD_VALUE_BEFORE_COLUMN]
        ca_a = data[CHILD_VALUE_AFTER_COLUMN]
        pb_a = data[PARENT_VALUE_BEFORE_COLUMN]

        data_norm = self.aggregate(indicator_norm)
        cb_b = data_norm[CHILD_VALUE_BEFORE_COLUMN]
        ca_b = data_norm[CHILD_VALUE_AFTER_COLUMN]
        pb_b = data_norm[PARENT_VALUE_BEFORE_COLUMN]
        pa_b = data_norm[PARENT_VALUE_AFTER_COLUMN]

        return (pb_a * pb_b - cb_a * cb_b + ca_a * ca_b) / pa_b

    def _aggregate(self, indicator: IndicatorEnum, data: dict[str, float | int]) -> int | float:
        if indicator.meta.per == "capita":
            return self._aggregate_normalized(data, DemographicIndicator.POPULATION)
        if indicator.meta.per == "area":
            return self._aggregate_normalized(data, GeneralIndicator.AREA)
        return data[PARENT_VALUE_BEFORE_COLUMN] - data[CHILD_VALUE_BEFORE_COLUMN] + data[CHILD_VALUE_AFTER_COLUMN]

    def aggregate(self, indicator: IndicatorEnum):
        data = self.get(indicator)
        if PARENT_VALUE_AFTER_COLUMN not in data:
            parent_after = self._aggregate(indicator, data)
            data[PARENT_VALUE_AFTER_COLUMN] = parent_after
        return data


# class SocioEconomicAggregator(_SocioEconomicAggregator):
#     """Class to operate with child and parent territory indicators values in case of effects"""

#     def _aggregate(self, parent_before: pd.Series, child_before: pd.Series, child_after: pd.Series) -> pd.Series:
#         return parent_before - child_before + child_after

#     def aggregate(self) -> pd.DataFrame:
#         if len(self._data) == 0:
#             raise ValueError("Cannot make DataFrame as no data was added")
#         if len(self._data) < len(self.indicator_cls):
#             logger.warning(f"Only {len(self._data)}/{len(self.indicator_cls)} indicators were added")

#         df = pd.DataFrame.from_dict(self._data, orient="index")
#         df[PARENT_VALUE_AFTER_COLUMN] = self._aggregate(
#             df[PARENT_VALUE_BEFORE_COLUMN], df[CHILD_VALUE_BEFORE_COLUMN], df[CHILD_VALUE_AFTER_COLUMN]
#         )
#         return df
