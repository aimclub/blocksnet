import geopandas as gpd
import pandas as pd
import shapely
from pydantic import BaseModel, InstanceOf

from .service_type import ServiceType


class Territory(BaseModel):
    id: int
    name: str
    geometry: InstanceOf[shapely.Polygon]

    @staticmethod
    def _aggregate_provision(prov_gdf: gpd.GeoDataFrame):
        columns = ["demand", "demand_left", "demand_within", "demand_without", "capacity", "capacity_left"]
        result = {column: prov_gdf[column].sum() for column in columns}
        result["provision"] = result["demand_within"] / result["demand"]
        return result

    def get_context_provision(
        self, service_type: ServiceType, towns_gdf: gpd.GeoDataFrame, speed: float = 283.33
    ) -> tuple[gpd.GeoDataFrame, dict[str, float]]:

        buffer_meters = service_type.accessibility * speed

        towns_gdf = towns_gdf.copy()
        towns_gdf["distance"] = towns_gdf["geometry"].apply(lambda g: shapely.distance(g, self.geometry))
        towns_gdf = towns_gdf[towns_gdf["distance"] <= buffer_meters]
        towns_gdf = towns_gdf.rename(columns={"index_right": "town_id"})

        territory_dict = self._aggregate_provision(towns_gdf)
        territory_dict["buffer_meters"] = buffer_meters
        return towns_gdf, territory_dict

    def get_indicators(self, provs: dict[ServiceType, tuple], min_provision = 0.8):
        dicts = []
        for service_type, prov_tuple in provs.items():
            _, _, towns_gdf, _ = prov_tuple
            _, territory_dict = self.get_context_provision(service_type, towns_gdf)
            dicts.append(
                {
                    "category": service_type.category.name,
                    "infrastructure": service_type.infrastructure.name,
                    "service_type": service_type.name,
                    "weight": service_type.weight,
                    **territory_dict,
                }
            )
        indicators = pd.DataFrame(dicts).set_index("service_type", drop=True)
        indicators["assessment"] = indicators.apply(lambda s: s["weight"] if s["provision"] >= min_provision else 0, axis=1)
        basic, basic_plus, comfort = indicators.groupby("category").agg({"assessment": "sum"})["assessment"]
        return indicators, basic + basic_plus/4 + comfort/4

    def to_dict(self):
        return {"id": self.id, "name": self.name, "geometry": self.geometry}

    @classmethod
    def from_gdf(cls, gdf: gpd.GeoDataFrame) -> list:
        return {i: cls(id=i, **gdf.loc[i].to_dict()) for i in gdf.index}
