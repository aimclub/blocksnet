from pydantic import BaseModel
from pulp import *
from itertools import product
from ...models import CityModel
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import math
import pandas as pd
import numpy as np


class LpProvision(BaseModel):
    city_model: CityModel
    services: dict = {
        "kindergartens": {"demand": 61, "accessibility": 10},
        "schools": {"demand": 120, "accessibility": 15},
        "recreational_areas": {"demand": 6000, "accessibility": 15},
        "hospitals": {"demand": 9, "accessibility": 60},
        "pharmacies": {"demand": 50, "accessibility": 10},
        "policlinics": {"demand": 27, "accessibility": 15},
    }

    def visualize_provision(self, provision, updated_blocks={}):
        ...

    def visualize_provisions(self, provisions: dict, updated_blocks={}):
        def show_me_chart(fig, gs, prov, name, i, sum):
            ax = fig.add_subplot(gs[i // 3, i % 3])
            self.city_model.blocks.to_gdf().plot(ax=ax, color="#ddd", alpha=1)
            prov.plot(column="provision", legend=True, ax=ax, cmap="RdYlGn", vmin=0, vmax=1)
            ax.set_title(name + " provision: " + f"{sum: .3f}")

        fig = plt.figure(figsize=(25, 15))
        gs = GridSpec(len(provisions) // 3 + 1, 3, figure=fig)
        i = 0
        for service_type, provision in provisions.items():
            show_me_chart(fig, gs, provision, service_type, i, self.sum_provision(provision))
            i = i + 1
        plt.show()

    def get_scenario_provisions(self, service_types, updated_blocks={}):
        provisions = {}
        sum = 0
        for service_type in service_types:
            provision = self.get_provision(service_type, updated_blocks)
            provisions[service_type] = provision
            sum += self.sum_provision(provision)
        return provisions, sum / len(provisions)

    @classmethod
    def sum_provision(cls, gdf):
        return gdf["supplied"].sum() / gdf["demand"].sum()

    def get_provision(self, service_type_name, updated_blocks={}):
        acc_df = self.city_model.accessibility_matrix.df
        blocks = self.city_model.blocks.to_gdf()
        for block_id, updated_info in updated_blocks.items():
            if "population" in updated_info:
                blocks.loc[block_id, "current_population"] += updated_info["population"]
        blocks["demand"] = (blocks["current_population"] / 1000 * self.services[service_type_name]["demand"]).apply(
            lambda x: math.ceil(x)
        )
        costs = pd.DataFrame(
            data=acc_df
        )  # .applymap(lambda x : math.exp(x) / self.services[service_type_name]['accessibility'])
        result = pd.DataFrame(index=costs.index, columns=costs.columns)
        demand = blocks["demand"]
        capacity = (
            pd.DataFrame.from_dict(self.city_model.services_graph.nodes, orient="index").fillna(0)[
                f"{service_type_name}_capacity"
            ]
        ).apply(lambda x: math.ceil(x))
        for block_id, updated_info in updated_blocks.items():
            if service_type_name in updated_info.items():
                capacity.loc[block_id] += updated_info[service_type_name]
        # drop 0 demand
        blocks.drop(labels=list(demand.loc[lambda x: x == 0].index), inplace=True, axis="index")
        costs.drop(labels=list(demand.loc[lambda x: x == 0].index), inplace=True, axis="index")
        demand = demand.loc[lambda x: x > 0]
        # drop 0 capacity
        costs.drop(labels=capacity.loc[lambda x: x == 0].index, inplace=True, axis="columns")
        capacity = capacity.loc[lambda x: x > 0]
        # add fictive blocks to balance the problem
        delta = demand.sum() - capacity.sum()
        last_index = None
        last_column = None
        if delta > 0:
            last_column = costs.iloc[:, -1].name + 1
            costs.loc[:, last_column + 1] = 0
            capacity[last_column + 1] = delta
        if delta < 0:
            last_index = costs.iloc[-1, :].name + 1
            costs.loc[last_index + 1, :] = 0
            demand[last_index + 1] = -delta
        # begin to solve the problem
        prob = LpProblem("Transportation", LpMinimize)
        x = LpVariable.dicts("Route", product(demand.index, capacity.index), 0, None)
        prob += lpSum(costs.loc[n, m] * x[n, m] for n in demand.index for m in capacity.index)
        for n in demand.index:
            prob += lpSum(x[n, m] for m in capacity.index) == demand[n]
        for m in capacity.index:
            prob += lpSum(x[n, m] for n in demand.index) == capacity[m]
        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        # make the output
        for var in prob.variables():
            value = var.value()
            name = var.name.replace("(", "").replace(")", "").replace(",", "").split("_")
            a = int(name[1])
            b = int(name[2])
            if value > 0 and costs.loc[a, b] <= self.services[service_type_name]["accessibility"]:
                if last_index != None and a != (last_index + 1):
                    result.loc[a, b] = value
                if last_column != None and b != (last_column + 1):
                    result.loc[a, b] = value
        blocks["demand"] = demand
        blocks["supplied"] = result.sum(axis=1)
        blocks["provision"] = blocks["supplied"] / blocks["demand"]
        return blocks
