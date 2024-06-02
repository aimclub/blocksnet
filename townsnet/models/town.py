import geopandas as gpd
import pandas as pd
import pyproj
import shapely
from pydantic import BaseModel, Field, InstanceOf, field_validator

from ..utils.basic_service_types import BASIC_SERVICE_TYPES
from .service_type import ServiceType


class Service(BaseModel):
    service_type: ServiceType
    geometry: InstanceOf[shapely.Point]
    capacity: int

    def to_dict(self):
        return {"service_type": self.service_type.name, "geometry": self.geometry, "capacity": self.capacity}

    @classmethod
    def from_gdf(cls, service_type: ServiceType, gdf) -> list:
        return [cls(service_type=service_type, **gdf.loc[i].to_dict()) for i in gdf.index]


class Town(BaseModel):
    id: int
    name: str
    population: int = Field(gt=0)
    geometry: InstanceOf[shapely.Point]
    services: list[Service] = []

    def __contains__(self, service_type: ServiceType) -> bool:
        """Returns True if service type is contained in town"""
        return service_type in self._capacities

    def __getitem__(self, service_type: ServiceType) -> dict[str, int]:
        """Get service type capacity and demand of the town"""
        result = {"capacity": 0, "demand": service_type.calculate_in_need(self.population)}
        if service_type in self._capacities:
            result["capacity"] = self._capacities[service_type]
        return result

    def to_dict(self):
        res = {"id": self.id, "name": self.name, "population": self.population, "geometry": self.geometry}
        df = pd.DataFrame([service.to_dict() for service in self.services])
        if not df.empty:
            for service_type, group in df.groupby("service_type"):
                res.update(
                    {
                        f"capacity_{service_type}": group["capacity"].sum(),
                        # f'count_{service_type}': len(group)
                    }
                )
        return res

    @classmethod
    def from_gdf(cls, gdf) -> dict:
        return {i: cls(id=i, **gdf.loc[i].to_dict()) for i in gdf.index}

    def update_services(self, service_type: ServiceType, gdf: gpd.GeoDataFrame | None = None):
        services = filter(lambda s: s.service_type != service_type, self.services)
        if gdf is not None:
            new_services = Service.from_gdf(service_type, gdf)
            services = [*services, *new_services]
        self.services = list(services)
