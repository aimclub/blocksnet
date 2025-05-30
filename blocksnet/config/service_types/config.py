from functools import singledispatchmethod

import pandas as pd

from ...enums import LandUse
from .common import SERVICE_TYPES, UNITS
from .schemas import LandUseSchema, ServiceTypesSchema, UnitsSchema


class ServiceTypesConfig:
    def __init__(self, service_types: pd.DataFrame, units: pd.DataFrame, land_use: pd.DataFrame):
        self.service_types = ServiceTypesSchema(service_types)
        self.units = UnitsSchema(units)
        self.land_use = LandUseSchema(land_use)

    @singledispatchmethod
    def __getitem__(self, arg):
        raise NotImplementedError(f"Cant access object with such argument type : {type(arg)}")

    @__getitem__.register(str)
    def _(self, service_type: str) -> dict:
        result = self.service_types.loc[service_type].to_dict()
        return result

    @__getitem__.register(LandUse)
    def _(self, land_use: LandUse) -> list[str]:
        row = self.land_use[land_use]
        service_types = row[row].index
        return list(service_types)

    def __iter__(self):
        return iter(self.service_types.index)

    def get_service_type(self, service_type: str) -> dict:
        row = self.service_types.loc[service_type]
        return row.to_dict()


service_types_config = ServiceTypesConfig(SERVICE_TYPES, UNITS, SERVICE_TYPES)
