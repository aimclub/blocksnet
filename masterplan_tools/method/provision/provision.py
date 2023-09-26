import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
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

    @classmethod
    def plot_delta(cls, gdf_before: gpd.GeoDataFrame, gdf_after: gpd.GeoDataFrame, column: str = "provision"):
        gdf = gdf_after.copy()
        gdf = gdf.loc[gdf[column] != 0]
        gdf[column] -= gdf_before[column]
        gdf.plot(column=column, cmap="RdYlGn", vmin=-1, vmax=1, legend=True).set_axis_off()

    @classmethod
    def plot_provisions(cls, provisions: dict[str, gpd.GeoDataFrame]):
        def show_me_chart(fig, gs, gdf, name, i, sum):
            ax = fig.add_subplot(gs[i // 3, i % 3])
            # self.city_model.blocks.to_gdf().plot(ax=ax, color="#ddd", alpha=1)
            gdf.plot(column="provision", legend=True, ax=ax, cmap="RdYlGn", vmin=0, vmax=1)
            ax.set_title(name + " provision: " + f"{sum: .3f}")
            ax.set_axis_off()

        fig = plt.figure(figsize=(25, 15))
        gs = GridSpec(len(provisions) // 3 + 1, 3, figure=fig)
        i = 0
        for service_type, provision_gdf in provisions.items():
            show_me_chart(fig, gs, provision_gdf, service_type, i, cls.total_provision(provision_gdf))
            i = i + 1
        plt.show()

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

    @classmethod
    def total_provision(cls, gdf: gpd.GeoDataFrame):
        return gdf["demand_within"].sum() / gdf["demand"].sum()

    @classmethod
    def mean_provision(cls, gdf: gpd.GeoDataFrame):
        return gdf["provision"].mean()

    def calculate_provisions(
        self, service_types: list[ServiceType | str], update_df: pd.DataFrame = None
    ) -> dict[str, gpd.GeoDataFrame]:
        result = {}
        for service_type in service_types:
            result[service_type] = self.calculate_provision(service_type, update_df)
        return result

    def calculate_provision(self, service_type: ServiceType | str, update_df: pd.DataFrame = None) -> gpd.GeoDataFrame:
        """Provision assessment method for the current city and service type, can be used with certain updated blocks GeoDataFrame"""
        if not isinstance(service_type, ServiceType):
            service_type = self.city_model[service_type]

        demand_blocks = self._get_filtered_blocks(service_type, "demand")
        capacity_blocks = self._get_filtered_blocks(service_type, "capacity")
        gdf = self._blocks_gdf(service_type)

        if update_df is not None:
            gdf = gdf.join(update_df)
            gdf[update_df.columns] = gdf[update_df.columns].fillna(0)
            gdf["demand"] += gdf["population"].apply(service_type.calculate_in_need)
            gdf["demand_left"] = gdf["demand"]
            gdf["capacity"] += gdf[service_type.name]
            gdf["capacity_left"] += gdf[service_type.name]

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

        gdf["provision"] = gdf["demand_within"] / gdf["demand"]
        return gdf
