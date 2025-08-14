from enum import Enum, unique


@unique
class EngineeringIndicator(Enum):
    NON_GASIFIED_SETTLEMENTS = "non_gasified_settlements"
    INFRASTRUCTURE_OBJECT = "infrastructure_object"
    POWER_PLANT = "power_plant"
    WATER_INTAKE = "water_intake"
    TREATMENT_FACILITY = "treatment_facility"
    RESERVOIR = "reservoir"
    GAS_DISTRIBUTION = "gas_distribution"
