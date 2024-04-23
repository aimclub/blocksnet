import math
from itertools import product
from typing import Literal

import geopandas as gpd
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import pandas as pd
from pulp import PULP_CBC_CMD, LpMinimize, LpProblem, LpVariable, lpSum, LpInteger

from ...models import Block, ServiceType
from ..base_method import BaseMethod
from enum import Enum

PLOT_KWARGS = {"column": "provision", "cmap": "RdYlGn", "vmin": 0, "vmax": 1, "legend": True}


class ProvisionMethod(Enum):
    GREEDY = "greedy"
    GRAVITATIONAL = "gravitational"
    LINEAR = "linear"


class Provision(BaseMethod):
    """Class provides methods for service type provision assessment"""

    def plot(self, gdf: gpd.GeoDataFrame, figsize: tuple[int, int] = (10, 10)):
        """Visualizes provision assessment results"""
        ax = gdf.plot(color="#ddd", figsize=figsize)
        gdf.plot(ax=ax, **PLOT_KWARGS)
        ax.set_title("Provision: " + f"{self.total_provision(gdf): .3f}")
        ax.set_axis_off()

    def plot_delta(
        self, gdf_before: gpd.GeoDataFrame, gdf_after: gpd.GeoDataFrame, figsize: tuple[int, int] = (10, 10)
    ):
        fig = plt.figure(figsize=figsize)
        grid = GridSpec(3, 2)
        axes = {
            "delta": fig.add_subplot(grid[0:2, 0:2]),
            "before": fig.add_subplot(grid[2, 0]),
            "after": fig.add_subplot(grid[2, 1]),
        }
        for ax in axes.values():
            gdf_before.plot(ax=ax, color="#ddd")
            ax.set_axis_off()
        gdf_before.plot(ax=axes["before"], **PLOT_KWARGS)
        gdf_after.plot(ax=axes["after"], **PLOT_KWARGS)
        gdf_delta = gdf_after.copy()
        gdf_delta["provision"] = gdf_after["provision"] - gdf_before["provision"]
        gdf_delta.loc[gdf_delta["provision"] != 0].plot(
            column="provision", ax=axes["delta"], cmap="PuOr", vmin=-1, vmax=1, legend=True
        )
        prov_delta = self.total_provision(gdf_after) - self.total_provision(gdf_before)
        axes["delta"].set_title("Provision delta: " + f"{prov_delta: .3f}")

    def plot_provisions(self, provisions: dict[str, gpd.GeoDataFrame]):
        def show_me_chart(ax, gdf, name, sum):
            gdf.plot(column="provision", legend=True, ax=ax, cmap="RdYlGn", vmin=0, vmax=1)
            gdf[gdf["demand"] == 0].plot(ax=ax, color="#ddd", alpha=1)
            ax.set_title(str(name.name) + " provision: " + f"{sum: .3f}")
            ax.set_axis_off()

        n_plots = len(provisions)
        _, axs = plt.subplots(nrows=n_plots // 2 + n_plots % 2, ncols=2, figsize=(20, 20))
        for i, (service_type, provision_gdf) in enumerate(provisions.items()):
            show_me_chart(axs[i // 2][i % 2], provision_gdf, service_type, self.total_provision(provision_gdf))
        plt.show()

    def _get_blocks_gdf(self, service_type: ServiceType) -> gpd.GeoDataFrame:
        """Returns blocks gdf for provision assessment"""
        capacity_column = f"capacity_{service_type.name}"
        gdf = self.city_model.get_blocks_gdf()[["geometry", "population", capacity_column]].fillna(0)
        gdf["population"] = gdf["population"].apply(service_type.calculate_in_need)
        gdf = gdf.rename(columns={"population": "demand", capacity_column: "capacity"})
        gdf["capacity_left"] = gdf["capacity"]
        gdf["demand_left"] = gdf["demand"]
        gdf["demand_within"] = 0
        gdf["demand_without"] = 0
        return gdf

    @classmethod
    def stat_provision(cls, gdf: gpd.GeoDataFrame):
        return {
            "mean": gdf["provision"].mean(),
            "median": gdf["provision"].median(),
            "min": gdf["provision"].min(),
            "max": gdf["provision"].max(),
        }

    @classmethod
    def total_provision(cls, gdf: gpd.GeoDataFrame) -> float:
        return gdf["demand_within"].sum() / gdf["demand"].sum()

    def calculate_scenario(
        self,
        scenario: dict[str, float],
        update_df: pd.DataFrame | None = None,
        method: ProvisionMethod = ProvisionMethod.GRAVITATIONAL,
        self_supply: bool = False,
    ) -> tuple[dict[str, gpd.GeoDataFrame], float]:
        result = {}
        total: float = 0
        for service_type, weight in scenario.items():
            prov_gdf = self.calculate(service_type, update_df, method, self_supply)
            result[service_type] = prov_gdf
            total += weight * self.total_provision(prov_gdf)
        return result, total

    @staticmethod
    def _get_lower_bound(gdf):
        gdf = gdf.copy()
        gdf["demand_within"] = gdf.apply(lambda x: min(x["capacity"], x["demand"]), axis=1)
        return gdf["demand_within"].sum() / gdf["demand"].sum()

    @staticmethod
    def _get_upper_bound(gdf):
        gdf = gdf.copy()
        capacity = gdf["capacity"].sum()
        demand = gdf["demand"].sum()
        return min(capacity / demand, 1)

    def get_bounds(self, service_type: ServiceType | str):
        service_type: ServiceType = self.city_model[service_type]
        gdf = self._get_blocks_gdf(service_type)
        lower_bound = self._get_lower_bound(gdf)
        upper_bound = self._get_upper_bound(gdf)
        return lower_bound, upper_bound

    def calculate(
        self,
        service_type: ServiceType | str,
        update_df: pd.DataFrame | None = None,
        method: ProvisionMethod = ProvisionMethod.GRAVITATIONAL,
        self_supply: bool = False,
    ) -> gpd.GeoDataFrame:
        """Provision assessment using certain method for the current city and
        service type, can be used with certain updated blocks DataFrame"""
        if not isinstance(service_type, ServiceType):
            service_type: ServiceType = self.city_model[service_type]
        gdf = self._get_blocks_gdf(service_type)

        if update_df is not None:
            gdf = gdf.join(update_df)
            gdf[update_df.columns] = gdf[update_df.columns].fillna(0)
            if not "population" in gdf:
                gdf["population"] = 0
            gdf["demand"] += gdf["population"].apply(service_type.calculate_in_need)
            gdf["demand_left"] = gdf["demand"]
            if not service_type.name in gdf:
                gdf[service_type.name] = 0
            gdf["capacity"] += gdf[service_type.name]
            gdf["capacity_left"] += gdf[service_type.name]

        if self_supply:
            supply: pd.Series = gdf.apply(lambda x: min(x["demand"], x["capacity"]), axis=1)
            gdf["demand_within"] += supply
            gdf["demand_left"] -= supply
            gdf["capacity_left"] -= supply

        if method == ProvisionMethod.GREEDY:
            gdf = self._greedy_provision(gdf, service_type)
        else:
            gdf = self._tp_provision(gdf, service_type, method)

        gdf["provision"] = gdf["demand_within"] / gdf["demand"]
        return gdf

    def _tp_provision(self, gdf: gpd.GeoDataFrame, service_type: ServiceType, method: ProvisionMethod):
        """Transport problem assessment method"""

        # if delta is not 0, we add a dummy city block
        delta = gdf["demand"].sum() - gdf["capacity"].sum()
        fictive_demand = None
        fictive_capacity = None
        fictive_block_id = gdf.index.max() + 1
        if delta > 0:
            fictive_capacity = fictive_block_id
            gdf.loc[fictive_capacity, "capacity"] = delta
            gdf.loc[fictive_capacity, "capacity_left"] = delta
        if delta < 0:
            fictive_demand = fictive_block_id
            gdf.loc[fictive_demand, "demand"] = -delta
            gdf.loc[fictive_demand, "demand_left"] = -delta

        def _get_distance(id1: int, id2: int):
            if id1 == fictive_block_id or id2 == fictive_block_id:
                return 0
            block1 = self.city_model[id1]
            block2 = self.city_model[id2]
            distance = self.city_model.get_distance(block1, block2)
            return distance

        def _get_weight(id1: int, id2: int):
            distance = _get_distance(id1, id2)
            if method == ProvisionMethod.LINEAR:
                return distance
            return distance * distance

        demand = gdf.loc[gdf["demand_left"] > 0]
        capacity = gdf.loc[gdf["capacity_left"] > 0]

        prob = LpProblem("Transportation", LpMinimize)
        x = LpVariable.dicts("Route", product(demand.index, capacity.index), 0, None, cat=LpInteger)
        prob += lpSum(_get_weight(n, m) * x[n, m] for n, m in product(demand.index, capacity.index))
        for n in demand.index:
            prob += lpSum(x[n, m] for m in capacity.index) == demand.loc[n, "demand_left"]
        for m in capacity.index:
            prob += lpSum(x[n, m] for n in demand.index) == capacity.loc[m, "capacity_left"]
        prob.solve(PULP_CBC_CMD(msg=False))

        for var in prob.variables():
            value = var.value()
            name = var.name.replace("(", "").replace(")", "").replace(",", "").split("_")
            if name[2] == "dummy":
                continue
            a = int(name[1])
            b = int(name[2])
            distance = _get_distance(a, b)
            if value > 0:
                if a != fictive_demand and b != fictive_capacity:
                    if distance <= service_type.accessibility:
                        gdf.loc[a, "demand_within"] += value
                    else:
                        gdf.loc[a, "demand_without"] += value
                    gdf.loc[a, "demand_left"] -= value
                    gdf.loc[b, "capacity_left"] -= value

        if fictive_block_id is not None:
            gdf = gdf.drop(labels=[fictive_block_id])

        return gdf

    def _greedy_provision(self, gdf: gpd.GeoDataFrame, service_type: ServiceType):
        """Iterative provision assessment method"""

        demand_gdf = gdf.loc[gdf["demand_left"] > 0]
        capacity_gdf = gdf.loc[gdf["capacity_left"] > 0]

        demand_blocks = [self.city_model[i] for i in demand_gdf.index]
        capacity_blocks = [self.city_model[i] for i in capacity_gdf.index]

        while gdf["demand_left"].sum() > 0 and gdf["capacity_left"].sum() > 0:

            print(gdf["demand_left"].sum(), gdf["capacity_left"].sum())

            for demand_block in demand_blocks:
                if len(capacity_blocks) == 0:
                    break
                capacity_block = min(
                    capacity_blocks, key=lambda capacity_block: self.city_model[demand_block, capacity_block]
                )

                gdf.loc[demand_block.id, "demand_left"] -= 1
                distance = self.city_model[demand_block, capacity_block]
                if distance <= service_type.accessibility:
                    gdf.loc[demand_block.id, "demand_within"] += 1
                else:
                    gdf.loc[demand_block.id, "demand_without"] += 1
                # remove block if it's demand_left is empty
                if gdf.loc[demand_block.id, "demand_left"] == 0:
                    demand_blocks.remove(demand_block)

                gdf.loc[capacity_block.id, "capacity_left"] -= 1
                # remove block if its capacity_left is empty
                if gdf.loc[capacity_block.id, "capacity_left"] == 0:
                    capacity_blocks.remove(capacity_block)

        return gdf
