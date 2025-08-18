from enum import unique
from ..enum import IndicatorEnum
from ..meta import IndicatorMeta


@unique
class DemographicIndicator(IndicatorEnum):
    POPULATION = IndicatorMeta("population")
    DENSITY = IndicatorMeta("density", "area")
    BIRTH_RATE = IndicatorMeta("birth_rate", "capita")
    DEATH_RATE = IndicatorMeta("death_rate", "capita")
    MIGRATION_RATE = IndicatorMeta("migration_rate", "capita")
