from enum import unique
from ..indicator_enum import IndicatorEnum
from ..indicator_meta import IndicatorMeta


@unique
class GeneralIndicator(IndicatorEnum):
    """General socio-economic indicators available for aggregation."""
    AREA = IndicatorMeta("area", unit="km2")
    URBANIZATION = IndicatorMeta("urbanization", per="area")
