import pandas as pd
from .common import SERVICE_TYPES, UNITS
from .schemas import ServiceTypesSchema, UnitsSchema


class ServiceTypesConfig:
    def __init__(self, service_types: pd.DataFrame, units: pd.DataFrame):
        self.set_service_types(service_types)
        self.set_units(units)
        # self.service_types = ServiceTypesSchema(service_types)
        # self.units = UnitsSchema(service_types)

    def set_service_types(self, service_types: pd.DataFrame):
        self.service_types = ServiceTypesSchema(service_types)

    def set_units(self, units: pd.DataFrame):
        self.units = UnitsSchema(units)

    # def add_service_type(self, )


service_types_config = ServiceTypesConfig(SERVICE_TYPES, UNITS)
