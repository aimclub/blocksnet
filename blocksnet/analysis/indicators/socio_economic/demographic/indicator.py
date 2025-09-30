from enum import unique
from ..indicator_enum import IndicatorEnum
from ..indicator_meta import IndicatorMeta


@unique
class DemographicIndicator(IndicatorEnum):
    """Demographic indicators measuring population composition and change."""
    POPULATION = IndicatorMeta("population")
    DENSITY = IndicatorMeta("density", per="area", unit="people/km2")
    BIRTH_RATE = IndicatorMeta("birth_rate", per="capita")
    DEATH_RATE = IndicatorMeta("death_rate", per="capita")
    MIGRATION_RATE = IndicatorMeta("migration_rate", per="capita")
