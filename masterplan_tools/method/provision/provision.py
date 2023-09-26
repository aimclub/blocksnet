import geopandas as gpd
from typing import Literal
from pydantic import InstanceOf
from ..base_method import BaseMethod
from ...models import Block, ServiceType


class Provision(BaseMethod):
    """Class provides methods for service type provision assessment"""

    @classmethod
    def plot(cls, gdf: gpd.GeoDataFrame):
        """Visualizes provision assessment results"""
        gdf.plot(column="provision", cmap="RdYlGn", vmin=0, vmax=1, legend=True).set_axis_off()

    def _get_filtered_blocks(self, service_type: ServiceType, type: Literal["demand", "capacity"]) -> list[Block]:
        """Get blocks filtered by demand or capacity greater than 0"""
        return list(filter(lambda b: b[service_type.name][type] > 0, self.city_model.blocks))

    def _get_sorted_neighbors(self, block, capacity_blocks: list[Block]):
        return sorted(capacity_blocks, key=lambda b: self.city_model.graph[block][b]["weight"])

    def _blocks_gdf(self, service_type: ServiceType) -> dict[Block, dict]:
        """Returns blocks gdf for provision assessment"""
        data: list[dict] = []
        for block in self.city_model.blocks:
            data.append({"id": block.id, "geometry": block.geometry, **block[service_type.name]})
        gdf = gpd.GeoDataFrame(data).set_index("id").set_crs(epsg=self.city_model.epsg)
        gdf["demand_left"] = gdf["demand"]
        gdf["demand_within"] = 0
        gdf["demand_without"] = 0
        gdf["capacity_left"] = gdf["capacity"]
        return gdf

    def calculate_provision(self, service_type: ServiceType | str) -> gpd.GeoDataFrame:
        if not isinstance(service_type, ServiceType):
            service_type = self.city_model[service_type]

        demand_blocks = self._get_filtered_blocks(service_type, "demand")
        capacity_blocks = self._get_filtered_blocks(service_type, "capacity")
        gdf = self._blocks_gdf(service_type)

        while len(demand_blocks) > 0 and len(capacity_blocks) > 0:
            for demand_block in demand_blocks:
                neighbors = self._get_sorted_neighbors(demand_block, capacity_blocks)
                if len(neighbors) == 0:
                    break
                capacity_block = neighbors[0]
                gdf.loc[demand_block.id, "demand_left"] -= 1
                weight = self.city_model.graph[demand_block][capacity_block]["weight"]
                if weight <= service_type.accessibility:
                    gdf.loc[demand_block.id, "demand_within"] += 1
                else:
                    gdf.loc[demand_block.id, "demand_without"] += 1
                if gdf.loc[demand_block.id, "demand_left"] == 0:
                    demand_blocks.remove(demand_block)
                gdf.loc[capacity_block.id, "capacity_left"] -= 1
                if gdf.loc[capacity_block.id, "capacity_left"] == 0:
                    capacity_blocks.remove(capacity_block)

        gdf["provision"] = gdf["demand_within"] / gdf["demand_without"]
        return gdf
