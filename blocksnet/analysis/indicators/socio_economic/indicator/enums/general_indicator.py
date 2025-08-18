from enum import unique
from ..indicator_enum import IndicatorEnum
from ..indicator_meta import IndicatorMeta


@unique
class GeneralIndicator(IndicatorEnum):
    AREA = IndicatorMeta("area")
    URBANIZATION = IndicatorMeta("urbanization", "area")
