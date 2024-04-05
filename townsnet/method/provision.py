from itertools import product
from typing import Literal
from shapely import LineString

from enum import Enum
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec
from pulp import PULP_CBC_CMD, LpMinimize, LpProblem, LpVariable, lpSum

from ..models.region import Town
from ..models.service_type import ServiceType
from .base_method import BaseMethod

PLOT_KWARGS = {
  'column' : 'provision',
  'cmap': 'RdYlGn',
  'vmin':0,
  'vmax':1,
  'legend':True
}

class ProvisionMethod(Enum):
    ITERATIVE = 'iterative',
    LINEAR = 'linear',
    GRAVITATIONAL = 'gravitational'

class Provision(BaseMethod):
    """Class provides methods for service type provision assessment"""

    @staticmethod
    def _plot_provision(ax, gdf):
        gdf.sort_values(by='provision').plot(
            ax=ax,
            zorder=1,
            **PLOT_KWARGS
        )  

    def _plot_rayons(self, ax, gdf):
        self._plot_provision(ax,gdf)
        ax.set_title('Rayons')

    def _plot_okrugs(self, ax, gdf):
        self._plot_provision(ax,gdf)
        ax.set_title('Okrugs')   

    def _plot_towns(self, ax, service_type, prov_gdf, weights_gdf):
        service_type_name = service_type.name
        total_provision = self.total_provision(prov_gdf)
        weights_gdf.sort_values(by='demand').plot(
            ax=ax, 
            zorder=0,
            column='demand', 
            cmap='cool', 
            alpha=1, 
            vmax=200, 
        )
        self._plot_provision(ax, prov_gdf)
        ax.set_title(f'Provision {service_type_name}: {total_provision: .3f}')

    def _aggregate_unit(self, groupby):
        res = groupby.agg({
            'capacity':'sum',
            'demand': 'sum',
            'demand_within': 'sum'
        })
        res['provision'] = res['demand_within']/res['demand']
        return res

    def aggregate_units(self, prov_gdf):
        okrugs_gdf = self._aggregate_unit(prov_gdf.groupby('okrug_name'))
        rayons_gdf = self._aggregate_unit(prov_gdf.groupby('rayon_name'))
        okrugs_gdf = self.region.okrugs.merge(okrugs_gdf, left_on='name', right_index=True)
        rayons_gdf = self.region.rayons.merge(rayons_gdf, left_on='name', right_index=True)
        return okrugs_gdf, rayons_gdf

    def plot(self, service_type, prov_gdf: gpd.GeoDataFrame, weights_gdf : gpd.GeoDataFrame, figsize=(15,15)):
        """Visualizes provision assessment results"""
        if not isinstance(service_type, ServiceType):
            service_type = self.region[service_type]
        fig = plt.figure(figsize=figsize)
        grid = GridSpec(3,2)
        axes = {
            'towns': fig.add_subplot(grid[0:2,0:2]),
            'okrugs': fig.add_subplot(grid[2,0]),
            'rayons': fig.add_subplot(grid[2,1]) 
        }
        for ax in axes.values():
            ax.set_axis_off()
        okrugs_gdf, rayons_gdf = self.aggregate_units(prov_gdf)
        self._plot_towns(axes['towns'], service_type, prov_gdf, weights_gdf)
        self._plot_okrugs(axes['okrugs'], okrugs_gdf)
        self._plot_rayons(axes['rayons'], rayons_gdf)      

    def _get_filtered_blocks(self, service_type: ServiceType, type: Literal["demand", "capacity"]) -> list[Town]:
        """Get blocks filtered by demand or capacity greater than 0"""
        return list(filter(lambda b: b[service_type][type] > 0, self.region.towns))

    def _get_sorted_neighbors(self, block, capacity_blocks: list[Town]):
        return sorted(capacity_blocks, key=lambda b: self.region[block, b])

    def _get_towns_gdf(self, service_type: ServiceType) -> gpd.GeoDataFrame:
        """Returns blocks gdf for provision assessment"""
        gdf = self.region.to_gdf()[[
            'name',
            'okrug_name',
            'rayon_name',
            'geometry',
            f'{service_type.name}_demand',
            f'{service_type.name}_capacity',
        ]].rename(columns={
            f'{service_type.name}_demand':'demand',
            f'{service_type.name}_capacity':'capacity'
        })
        gdf["capacity_left"] = gdf["capacity"]
        gdf["demand_left"] = gdf["demand"]
        gdf["demand_within"] = 0
        gdf["demand_without"] = 0
        return gdf

    @classmethod
    def total_provision(cls, gdf: gpd.GeoDataFrame):
        return gdf["demand_within"].sum() / gdf["demand"].sum()

    def _series_to_linestring(self, series):
            town_from = self.region[series.name[0]]
            town_to = self.region[series.name[1]]
            return LineString([town_from.geometry, town_to.geometry])

    def calculate(
        self,
        service_type: ServiceType | str,
        # update_df: pd.DataFrame = None,
        method: ProvisionMethod = ProvisionMethod.GRAVITATIONAL,
    ) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """
        Provision assessment using certain method for the current region and service type, 
        can be used with certain updated blocks GeoDataFrame
        """
        if not isinstance(service_type, ServiceType):
            service_type = self.region[service_type]
        gdf = self._get_towns_gdf(service_type)
        # if update_df is not None:
        #     gdf = gdf.join(update_df)
        #     gdf[update_df.columns] = gdf[update_df.columns].fillna(0)
        #     if not "population" in gdf:
        #         gdf["population"] = 0
        #     gdf["demand"] += gdf["population"].apply(service_type.calculate_in_need)
        #     gdf["demand_left"] = gdf["demand"]
        #     if not service_type.name in gdf:
        #         gdf[service_type.name] = 0
        #     gdf["capacity"] += gdf[service_type.name]
        #     gdf["capacity_left"] += gdf[service_type.name]
        match method:
            case ProvisionMethod.LINEAR:
                gdf, df = self._tp_provision(gdf, service_type, "lp")
            case ProvisionMethod.GRAVITATIONAL:
                gdf, df = self._tp_provision(gdf, service_type, "gravity")
            case ProvisionMethod.ITERATIVE:
                gdf, df = self._iterative_provision(gdf, service_type)
        
        gdf["provision"] = gdf["demand_within"] / gdf["demand"]
        df['geometry'] = df.apply(self._series_to_linestring,axis=1)
        df = gpd.GeoDataFrame(df).set_geometry('geometry').set_crs(self.region.crs)

        return gdf, df

    def _tp_provision(
        self, gdf: gpd.GeoDataFrame, service_type: ServiceType, type: Literal["lp", "gravity"]
    ):
        """Transport problem assessment method"""
        gdf = gdf.copy()

        delta = gdf["demand"].sum() - gdf["capacity"].sum()
        fictive_index = None
        fictive_column = None
        fictive_town_id = gdf.index.max() + 1
        if delta > 0:
            fictive_column = fictive_town_id
            gdf.loc[fictive_column, "capacity"] = delta
        if delta < 0:
            fictive_index = fictive_town_id
            gdf.loc[fictive_index, "demand"] = -delta
        gdf["capacity_left"] = gdf["capacity"]
        gdf['demand_left'] = gdf['demand']

        def _get_weight(town_a_id: int, town_b_id: int):
            if town_a_id == fictive_town_id or town_b_id == fictive_town_id:
                return 0
            town_a_id = self.region[town_a_id]
            town_b = self.region[town_b_id]
            distance = self.region[town_a_id, town_b]
            if type == "lp":
                return distance
            return distance * distance

        def _get_distance(town_a_id: int, town_b_id: int):
            if town_a_id == fictive_town_id or town_b_id == fictive_town_id:
                return 0
            town_a = self.region[town_a_id]
            town_b = self.region[town_b_id]
            distance = self.region[town_a, town_b]
            return distance

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
        edges = []
        # make the output
        for var in prob.variables():
            value = var.value()
            name = var.name.replace("(", "").replace(")", "").replace(",", "").split("_")
            if name[2] == "dummy":
                continue
            a = int(name[1])
            b = int(name[2])
            weight = _get_distance(a, b)
            if value > 0:
                if a!=fictive_town_id and b!=fictive_town_id:
                    edges.append({
                        'from': a,
                        'to': b,
                        'demand': value
                    })
                if weight <= service_type.accessibility:
                    if fictive_index != None and a != fictive_index:
                        gdf.loc[a, "demand_within"] += value
                        gdf.loc[b, "capacity_left"] -= value
                    if fictive_column != None and b != fictive_column:
                        gdf.loc[a, "demand_within"] += value
                        gdf.loc[b, "capacity_left"] -= value
                else:
                    if fictive_index != None and a != fictive_index:
                        gdf.loc[a, "demand_without"] += value
                        gdf.loc[b, "capacity_left"] -= value
                    if fictive_column != None and b != fictive_column:
                        gdf.loc[a, "demand_without"] += value
                        gdf.loc[b, "capacity_left"] -= value
        if fictive_index != None or fictive_column != None:
            gdf.drop(labels=[fictive_town_id], inplace=True)
        gdf["demand_left"] = gdf["demand"] - gdf["demand_within"] - gdf["demand_without"]
        return gdf, pd.DataFrame(edges).set_index(['from', 'to'])

    def _iterative_provision(self, gdf: gpd.GeoDataFrame, service_type: ServiceType):
        """Iterative provision assessment method"""

        demand_blocks = self._get_filtered_blocks(service_type, "demand")
        capacity_blocks = self._get_filtered_blocks(service_type, "capacity")

        edges = []

        while len(demand_blocks) > 0 and len(capacity_blocks) > 0:
            for demand_block in demand_blocks:
                neighbors = self._get_sorted_neighbors(demand_block, capacity_blocks)
                if len(neighbors) == 0:
                    break
                capacity_block = neighbors[0]
                gdf.loc[demand_block.id, "demand_left"] -= 1
                # df.loc[demand_block.id, capacity_block.id] += 1
                weight = self.region[demand_block, capacity_block]
                if weight <= service_type.accessibility:
                    gdf.loc[demand_block.id, "demand_within"] += 1
                else:
                    gdf.loc[demand_block.id, "demand_without"] += 1
                if gdf.loc[demand_block.id, "demand_left"] == 0:
                    demand_blocks.remove(demand_block)
                gdf.loc[capacity_block.id, "capacity_left"] -= 1
                if gdf.loc[capacity_block.id, "capacity_left"] == 0:
                    capacity_blocks.remove(capacity_block)

        return gdf, pd.DataFrame(edges)
