from enum import unique
from ..enum import IndicatorEnum
from ..meta import IndicatorMeta


@unique
class GeneralIndicator(IndicatorEnum):
    AREA = IndicatorMeta("area")
    URBANIZATION = IndicatorMeta("urbanization", "area")
