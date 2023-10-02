import math
import geopandas as gpd
import contextily as cx
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, PULP_CBC_CMD
from matplotlib.gridspec import GridSpec
from typing import Literal
from ..base_method import BaseMethod
from ...models import Block, ServiceType


class Provision(BaseMethod):
    """Class provides methods for service type provision assessment"""

    def _add_basemap(self, ax):
        cx.add_basemap(ax, source=cx.providers.Stamen.TonerLite, crs=f"EPSG:{self.city_model.epsg}")

    def plot(self, gdf: gpd.GeoDataFrame):
        """Visualizes provision assessment results"""
        ax = gdf.plot(column="provision", cmap="RdYlGn", vmin=0, vmax=1, legend=True).set_axis_off()
        self._add_basemap(ax)

    def plot_delta(self, gdf_before: gpd.GeoDataFrame, gdf_after: gpd.GeoDataFrame):
        gdf = gdf_after.copy()
        gdf = gdf.loc[gdf["provision"] != 0]
        gdf["provision"] -= gdf_before["provision"]
        ax = gdf.plot(column="provision", cmap="RdYlGn", vmin=-1, vmax=1, legend=True)
        ax.set_axis_off()
        self._add_basemap(ax)

    def plot_provisions(self, provisions: dict[str, gpd.GeoDataFrame]):
        def show_me_chart(fig, gs, gdf, name, i, sum):
            ax = fig.add_subplot(gs[i // 3, i % 3])
            # self.city_model.blocks.to_gdf().plot(ax=ax, color="#ddd", alpha=1)
            gdf.plot(column="provision", legend=True, ax=ax, cmap="RdYlGn", vmin=0, vmax=1)
            ax.set_title(name + " provision: " + f"{sum: .3f}")
            ax.set_axis_off()
            self._add_basemap(ax)

        fig = plt.figure(figsize=(25, 15))
        gs = GridSpec(len(provisions) // 3 + 1, 3, figure=fig)
        i = 0
        for service_type, provision_gdf in provisions.items():
            show_me_chart(fig, gs, provision_gdf, service_type, i, self.total_provision(provision_gdf))
            i = i + 1
        plt.show()

    def _get_filtered_blocks(self, service_type: ServiceType, type: Literal["demand", "capacity"]) -> list[Block]:
        """Get blocks filtered by demand or capacity greater than 0"""
        return list(filter(lambda b: b[service_type.name][type] > 0, self.city_model.blocks))

    def _get_sorted_neighbors(self, block, capacity_blocks: list[Block]):
        return sorted(capacity_blocks, key=lambda b: self.city_model.graph[block][b]["weight"])

    def _get_blocks_gdf(self, service_type: ServiceType) -> dict[Block, dict]:
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

    def calculate_provision(
        self, service_type: ServiceType | str, update_df: pd.DataFrame = None, method: Literal["iterative", "lp"] = "lp"
    ) -> gpd.GeoDataFrame:
        """Provision assessment using certain method for the current city and service type, can be used with certain updated blocks GeoDataFrame"""
        if not isinstance(service_type, ServiceType):
            service_type = self.city_model[service_type]
        gdf = self._get_blocks_gdf(service_type)
        if update_df is not None:
            gdf = gdf.join(update_df)
            gdf[update_df.columns] = gdf[update_df.columns].fillna(0)
            gdf["demand"] += gdf["population"].apply(service_type.calculate_in_need)
            gdf["demand_left"] = gdf["demand"]
            gdf["capacity"] += gdf[service_type.name]
            gdf["capacity_left"] += gdf[service_type.name]

        match method:
            case "lp":
                gdf = self._lp_provision(gdf, service_type)
            case "iterative":
                gdf = self._iterative_provision(gdf, service_type)
        gdf["provision"] = gdf["demand_within"] / gdf["demand"]

        return gdf

    def _lp_provision(self, gdf: gpd.GeoDataFrame, service_type: ServiceType) -> gpd.GeoDataFrame:
        """Linear programming assessment method"""
        gdf = gdf.copy()

        delta = gdf["demand"].sum() - gdf["capacity"].sum()
        fictive_index = None
        fictive_column = None
        fictive_block_id = gdf.index.max() + 1
        if delta > 0:
            fictive_column = fictive_block_id
            gdf.loc[fictive_column, "capacity"] = delta
        if delta < 0:
            fictive_index = fictive_block_id
            gdf.loc[fictive_index, "demand"] = -delta
        gdf["capacity_left"] = gdf["capacity"]

        def _get_weight(id1: int, id2: int):
            if id1 == fictive_index or id1 == fictive_column:
                return 0
            if id2 == fictive_index or id2 == fictive_column:
                return 0
            block1 = self.city_model[id1]
            block2 = self.city_model[id2]
            return int(self.city_model.graph[block1][block2]["weight"])

        demand = gdf.loc[gdf["demand"] > 0]
        capacity = gdf.loc[gdf["capacity"] > 0]

        prob = LpProblem("Transportation", LpMinimize)
        x = LpVariable.dicts("Route", product(demand.index, capacity.index), 0, None)
        prob += lpSum(_get_weight(n, m) * x[n, m] for n in demand.index for m in capacity.index)
        for n in demand.index:
            prob += lpSum(x[n, m] for m in capacity.index) == demand.loc[n, "demand"]
        for m in capacity.index:
            prob += lpSum(x[n, m] for n in demand.index) == capacity.loc[m, "capacity"]
        prob.solve(PULP_CBC_CMD(msg=False))
        # make the output
        for var in prob.variables():
            value = var.value()
            name = var.name.replace("(", "").replace(")", "").replace(",", "").split("_")
            a = int(name[1])
            b = int(name[2])
            weight = _get_weight(a, b)
            if value > 0:
                if weight <= service_type.accessibility:
                    if fictive_index != None and a != fictive_index:
                        gdf.loc[a, "demand_within"] = value
                        gdf.loc[b, "capacity_left"] = value
                    if fictive_column != None and b != fictive_column:
                        gdf.loc[a, "demand_within"] = value
                        gdf.loc[b, "capacity_left"] = value
                else:
                    if fictive_index != None and a != fictive_index:
                        gdf.loc[a, "demand_without"] = value
                        gdf.loc[b, "capacity_left"] = value
                    if fictive_column != None and b != fictive_column:
                        gdf.loc[a, "demand_without"] = value
                        gdf.loc[b, "capacity_left"] = value
        if fictive_index != None:
            gdf.drop(labels=[fictive_index], inplace=True)
        if fictive_column != None:
            gdf.drop(labels=[fictive_column], inplace=True)
        return gdf

    def _iterative_provision(self, gdf: gpd.GeoDataFrame, service_type: ServiceType) -> gpd.GeoDataFrame:
        """Iterative provision assessment method"""

        demand_blocks = self._get_filtered_blocks(service_type, "demand")
        capacity_blocks = self._get_filtered_blocks(service_type, "capacity")

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

        return gdf
