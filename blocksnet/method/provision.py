import math
from itertools import product
from typing import Literal
from loguru import logger

import geopandas as gpd
from matplotlib.gridspec import GridSpec
from multiprocessing import Pool
import matplotlib.pyplot as plt
import pandas as pd
from pulp import PULP_CBC_CMD, LpMinimize, LpMaximize, LpProblem, LpVariable, lpSum, LpInteger
from tqdm import tqdm
from ..models import Block, ServiceType
from .base_method import BaseMethod
from enum import Enum

PROVISION_COLUMN = "provision"

PLOT_KWARGS = {"column": PROVISION_COLUMN, "cmap": "RdYlGn", "vmin": 0, "vmax": 1, "legend": True}


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

    def plot_provisions(self, provisions: dict[str, gpd.GeoDataFrame], figsize=(20, 20)):

        n_plots = len(provisions)
        n_cols = 2
        n_rows = n_plots // n_cols + n_plots % n_cols

        fig = plt.figure(figsize=figsize)
        grid = GridSpec(n_rows, n_cols)

        for i, (service_type_name, provision_gdf) in enumerate(provisions.items()):
            ax = fig.add_subplot(grid[i // n_cols, i % n_cols])
            provision_gdf.plot(ax=ax, color="#ddd")
            provision_gdf.plot(ax=ax, **PLOT_KWARGS)
            total = self.total_provision(provision_gdf)
            ax.set_title(f"{service_type_name} provision: {total: .3f}")
            ax.set_axis_off()

    def _get_blocks_gdf(self, service_type: ServiceType, update_df: pd.DataFrame | None = None) -> gpd.GeoDataFrame:
        """Returns blocks gdf for provision assessment"""
        capacity_column = f"capacity_{service_type.name}"
        gdf = self.city_model.get_blocks_gdf()[["geometry", "population", capacity_column]].fillna(0)
        gdf = gdf.rename(columns={capacity_column: "capacity"})
        if update_df is not None:
            if "population" in update_df.columns:
                gdf["population"] = gdf["population"].add(update_df["population"].fillna(0), fill_value=0)
            if service_type.name in update_df.columns:
                gdf["capacity"] += gdf["capacity"].add(update_df[service_type.name].fillna(0), fill_value=0)
        gdf["population"] = gdf["population"].apply(service_type.calculate_in_need)
        gdf = gdf.rename(columns={"population": "demand"})
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

    def _calculate_provision(self, params):
        return self.calculate(**params)

    def calculate_scenario(
        self,
        scenario: dict[str, float],
        update_df: pd.DataFrame | None = None,
        method: ProvisionMethod = ProvisionMethod.GRAVITATIONAL,
        self_supply: bool = False,
    ) -> tuple[dict[str, gpd.GeoDataFrame], float]:

        with Pool() as pool:
            gdfs = [
                gdf
                for gdf in pool.map(
                    self._calculate_provision,
                    [
                        {
                            "service_type": service_type,
                            "update_df": update_df,
                            "method": method,
                            "self_supply": self_supply,
                        }
                        for service_type in scenario.keys()
                    ],
                )
            ]

        result = {}
        total = 0
        for i, gdf in enumerate(gdfs):
            service_type = list(scenario.keys())[i]
            weight = scenario[service_type]
            total_prov = self.total_provision(gdf)
            result[service_type] = gdf
            total += total_prov * weight

        # for service_type, weight in tqdm(scenario.items()):
        #     prov_gdf = self.calculate(service_type, update_df, method, self_supply)
        #     result[service_type] = prov_gdf
        #     total += weight * self.total_provision(prov_gdf)
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

    def get_bounds(self, service_type: ServiceType | str, update_df: pd.DataFrame = None):
        service_type: ServiceType = self.city_model[service_type]
        gdf = self._get_blocks_gdf(service_type, update_df)
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
        gdf = self._get_blocks_gdf(service_type, update_df)

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

        def _get_distance(id1: int, id2: int):
            distance = self.city_model.adjacency_matrix.loc[id1, id2]
            return distance if distance > 1 else 1

        def _get_weight(id1: int, id2: int):
            distance = _get_distance(id1, id2)
            if method == ProvisionMethod.LINEAR:
                return 1 / distance
            return 1 / (distance * distance)

        if self.verbose:
            logger.info("Setting an LP problem")

        demand = gdf.loc[gdf["demand_left"] > 0]
        capacity = gdf.loc[gdf["capacity_left"] > 0]

        prob = LpProblem("Provision", LpMaximize)
        # Precompute distance and filter products
        products = [
            (i, j)
            for i in demand.index
            for j in capacity.index
            if _get_distance(i, j) <= service_type.accessibility * 2
        ]

        # Create the decision variable dictionary
        x = LpVariable.dicts("Route", products, 0, None, cat=LpInteger)

        # Objective Function
        prob += lpSum(_get_weight(n, m) * x[n, m] for n, m in products)

        # Constraint dictionaries
        demand_constraints = {n: [] for n in demand.index}
        capacity_constraints = {m: [] for m in capacity.index}

        for n, m in products:
            demand_constraints[n].append(x[n, m])
            capacity_constraints[m].append(x[n, m])

        # Add Demand Constraints
        for n in demand.index:
            prob += lpSum(demand_constraints[n]) <= demand.loc[n, "demand_left"]

        # Add Capacity Constraints
        for m in capacity.index:
            prob += lpSum(capacity_constraints[m]) <= capacity.loc[m, "capacity_left"]

        if self.verbose:
            logger.info("Solving the problem")
        prob.solve(PULP_CBC_CMD(msg=False))

        if self.verbose:
            logger.info("Restoring values from variables")

        for var in prob.variables():
            value = var.value()
            name = var.name.replace("(", "").replace(")", "").replace(",", "").split("_")
            if name[2] == "dummy":
                continue
            a = int(name[1])
            b = int(name[2])
            distance = _get_distance(a, b)
            if value > 0:
                if distance <= service_type.accessibility:
                    gdf.loc[a, "demand_within"] += value
                else:
                    gdf.loc[a, "demand_without"] += value
                gdf.loc[a, "demand_left"] -= value
                gdf.loc[b, "capacity_left"] -= value

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
