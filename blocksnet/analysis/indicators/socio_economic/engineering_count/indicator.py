from enum import Enum, unique


@unique
class EngineeringCountIndicator(Enum):
    # education
    ...  # количество негазифицированных НП
    INFRASTRUCTURE_OBJECT = "infrastructure_object"
    POWER_PLANT = "power_plant"
    WATER_INTAKE = "water_intake"
    TREATMENT_FACILITY = "treatment_facility"
    RESERVOIR = "reservoir"
    GAS_DISTRIBUTION = "gas_distribution"
