from enum import unique
from ...indicator.enum import IndicatorEnum
from ...indicator.meta import IndicatorMeta


@unique
class EngineeringIndicator(IndicatorEnum):
    """EngineeringIndicator class.

    """
    NON_GASIFIED_SETTLEMENTS = IndicatorMeta("non_gasified_settlements")
    INFRASTRUCTURE_OBJECT = IndicatorMeta("infrastructure_object")
    POWER_PLANT = IndicatorMeta("power_plant")
    WATER_INTAKE = IndicatorMeta("water_intake")
    TREATMENT_FACILITY = IndicatorMeta("treatment_facility")
    RESERVOIR = IndicatorMeta("reservoir")
    GAS_DISTRIBUTION = IndicatorMeta("gas_distribution")
