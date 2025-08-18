from enum import unique
from ..indicator_enum import IndicatorEnum
from ..indicator_meta import IndicatorMeta


@unique
class EcologicalIndicator(IndicatorEnum):
    PROTECTED_AREA_SHARE = IndicatorMeta("protected_area_share", "area")
    URBAN_GREEN_AREA = IndicatorMeta("urban_green_area")
    NEUTRALIZED_POLLUTION_SHARE = IndicatorMeta("neutralized_pollution_share", aggregatable=False)
    ACCUMULATED_DAMAGE_AREA = IndicatorMeta("accumulated_damage_area")
    NEGATIVE_IMPACT_AREA_SHARE = IndicatorMeta("negative_impact_area_share", "area")
    POLLUTANTS_COUNT = IndicatorMeta("pollutants_count")
