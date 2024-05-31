from itertools import product
from typing import Literal
from shapely import LineString

from enum import Enum
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec
from pulp import PULP_CBC_CMD, LpMinimize, LpMaximize, LpProblem, LpVariable, lpSum, LpInteger

from ..models.region import Town
from ..models.service_type import ServiceType
from .base_method import BaseMethod

PROVISION_COLUMN = 'provision'
DEMAND_COLUMN = 'demand'
DEMAND_WITHIN_COLUMN = 'demand_within'
DEMAND_WITHOUT_COLUMN = 'demand_without'
DEMAND_LEFT_COLUMN = 'demand_left'
CAPACITY_COLUMN = 'capacity'
CAPACITY_LEFT_COLUMN = 'capacity_left'

PLOT_KWARGS = {
  'column' : PROVISION_COLUMN,
  'cmap': 'RdYlGn',
  'vmin':0,
  'vmax':1,
  'legend':True
}

class ProvisionMethod(Enum):
    GRAVITATIONAL = "gravitational"
    LINEAR = "linear"

class Provision(BaseMethod):
    """Class provides methods for service type provision assessment"""

    @staticmethod
    def _plot_provision(ax, gdf):
        gdf.sort_values(by=PROVISION_COLUMN).plot(
            ax=ax,
            zorder=1,
            **PLOT_KWARGS
        )

    def _plot_districts(self, ax, gdf):
        self._plot_provision(ax,gdf)
        ax.set_title('Districts')

    def _plot_settlements(self, ax, gdf):
        self._plot_provision(ax,gdf)
        ax.set_title('Settlements') 

    def _plot_towns(self, ax, service_type, prov_gdf, links_gdf):
        service_type_name = service_type.name
        total_provision = self.total_provision(prov_gdf)
        links_gdf.sort_values(by=DEMAND_COLUMN).plot(
            ax=ax,
            zorder=0,
            column=DEMAND_COLUMN,
            cmap='cool',
            alpha=1,
            vmax=200,
        )
        self._plot_provision(ax, prov_gdf)
        ax.set_title(f'Provision {service_type_name}: {total_provision: .3f}')

    def _aggregate_unit(self, groupby):
        res = groupby.agg({
            CAPACITY_COLUMN:'sum',
            CAPACITY_LEFT_COLUMN:'sum',
            DEMAND_COLUMN: 'sum',
            DEMAND_WITHIN_COLUMN: 'sum',
            DEMAND_WITHOUT_COLUMN: 'sum',
        })
        res[PROVISION_COLUMN] = res[DEMAND_WITHIN_COLUMN]/res[DEMAND_COLUMN]
        return res

    def _aggregate_units(self, prov_gdf) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        settlements_gdf = self._aggregate_unit(prov_gdf.groupby('settlement_name'))
        districts_gdf = self._aggregate_unit(prov_gdf.groupby('district_name'))
        settlements_gdf = self.region.settlements.merge(settlements_gdf, left_on='name', right_index=True)
        districts_gdf = self.region.districts.merge(districts_gdf, left_on='name', right_index=True)
        return settlements_gdf, districts_gdf

    def plot(self, service_type : str | ServiceType, districts_gdf : gpd.GeoDataFrame, settlements_gdf : gpd.GeoDataFrame, towns_gdf: gpd.GeoDataFrame, links_gdf : gpd.GeoDataFrame, figsize=(15,15)):
        """Visualizes provision assessment results"""
        if not isinstance(service_type, ServiceType):
            service_type = self.region[service_type]
        fig = plt.figure(figsize=figsize)
        grid = GridSpec(3,2)
        axes = {
            'towns': fig.add_subplot(grid[0:2,0:2]),
            'settlements': fig.add_subplot(grid[2,0]),
            'districts': fig.add_subplot(grid[2,1]) 
        }
        for ax in axes.values():
            ax.set_axis_off()
        self._plot_towns(axes['towns'], service_type, towns_gdf, links_gdf)
        self._plot_settlements(axes['settlements'], settlements_gdf)
        self._plot_districts(axes['districts'], districts_gdf)   

    def _get_towns_gdf(self, service_type: ServiceType) -> gpd.GeoDataFrame:
        """Returns towns gdf for provision assessment"""
        capacity_column = f"capacity_{service_type.name}"
        gdf = self.region.get_towns_gdf()[["geometry", 'town_name', 'settlement_name', 'district_name', "population", capacity_column]].fillna(0)
        gdf["population"] = gdf["population"].apply(service_type.calculate_in_need)
        gdf = gdf.rename(columns={"population": DEMAND_COLUMN, capacity_column: CAPACITY_COLUMN})
        gdf[CAPACITY_LEFT_COLUMN] = gdf[CAPACITY_COLUMN]
        gdf[DEMAND_LEFT_COLUMN] = gdf[DEMAND_COLUMN]
        gdf[DEMAND_WITHIN_COLUMN] = 0
        gdf[DEMAND_WITHOUT_COLUMN] = 0
        return gdf

    @classmethod
    def total_provision(cls, gdf: gpd.GeoDataFrame):
        return gdf[DEMAND_WITHIN_COLUMN].sum() / gdf[DEMAND_COLUMN].sum()

    def _link_to_linestring(self, dict):
        town_from = self.region[int(dict['from'])]
        town_to = self.region[int(dict['to'])]
        return LineString([town_from.geometry, town_to.geometry])

    def calculate(
        self,
        service_type: ServiceType | str,
        update_df: pd.DataFrame | None = None,
        method: ProvisionMethod = ProvisionMethod.GRAVITATIONAL,
        self_supply: bool = True,
    ) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """Provision assessment using certain method for the region and
        service type, can be used with certain updated capacity and population info df"""
        
        if not isinstance(service_type, ServiceType):
            service_type: ServiceType = self.region[service_type]
        towns_gdf = self._get_towns_gdf(service_type)

        if update_df is not None:
            towns_gdf = towns_gdf.join(update_df)
            towns_gdf[update_df.columns] = towns_gdf[update_df.columns].fillna(0)
            if not "population" in towns_gdf:
                towns_gdf["population"] = 0
            towns_gdf[DEMAND_COLUMN] += towns_gdf["population"].apply(service_type.calculate_in_need)
            towns_gdf[DEMAND_LEFT_COLUMN] = towns_gdf[DEMAND_COLUMN]
            if not service_type.name in towns_gdf:
                towns_gdf[service_type.name] = 0
            towns_gdf[CAPACITY_COLUMN] += towns_gdf[service_type.name]
            towns_gdf[CAPACITY_LEFT_COLUMN] += towns_gdf[service_type.name]

        if self_supply:
            supply: pd.Series = towns_gdf.apply(lambda x: min(x[DEMAND_COLUMN], x[CAPACITY_COLUMN]), axis=1)
            towns_gdf[DEMAND_WITHIN_COLUMN] += supply
            towns_gdf[DEMAND_LEFT_COLUMN] -= supply
            towns_gdf[CAPACITY_LEFT_COLUMN] -= supply

        towns_gdf, links = self._tp_provision(towns_gdf, service_type, method)

        for link in links:
            link['geometry'] = self._link_to_linestring(link)
        links_gdf = gpd.GeoDataFrame(links, columns=['geometry', 'from', 'to', 'demand', 'within']).set_crs(self.region.crs)

        towns_gdf[PROVISION_COLUMN] = towns_gdf[DEMAND_WITHIN_COLUMN] / towns_gdf[DEMAND_COLUMN]
        settlements_gdf, districts_gdf = self._aggregate_units(towns_gdf)
        return districts_gdf, settlements_gdf, towns_gdf, links_gdf

    def _tp_provision(self, gdf: gpd.GeoDataFrame, service_type: ServiceType, method: ProvisionMethod) -> tuple[gpd.GeoDataFrame, list[dict]]:
        """Transport problem assessment method"""

        links = []
        # if delta is not 0, we add a dummy city block
        # delta = gdf[DEMAND_COLUMN].sum() - gdf[CAPACITY_COLUMN].sum()
        # fictive_demand = None
        # fictive_capacity = None
        # fictive_town_id = gdf.index.max() + 1
        # if delta > 0:
        #     fictive_capacity = fictive_town_id
        #     gdf.loc[fictive_capacity, CAPACITY_COLUMN] = delta
        #     gdf.loc[fictive_capacity, CAPACITY_LEFT_COLUMN] = delta
        # if delta < 0:
        #     fictive_demand = fictive_town_id
        #     gdf.loc[fictive_demand, DEMAND_COLUMN] = -delta
        #     gdf.loc[fictive_demand, DEMAND_LEFT_COLUMN] = -delta

        def _get_distance(id1: int, id2: int):
            # if id1 == fictive_town_id or id2 == fictive_town_id:
                # return 0
            distance = self.region[id1, id2]
            return distance if distance>1 else 1

        def _get_weight(id1: int, id2: int):
            distance = _get_distance(id1, id2)
            if method == ProvisionMethod.GRAVITATIONAL:
                return 1 / (distance * distance)
            return 1/distance

        demand = gdf.loc[gdf[DEMAND_LEFT_COLUMN] > 0]
        capacity = gdf.loc[gdf[CAPACITY_LEFT_COLUMN] > 0]

        prob = LpProblem("Provision", LpMaximize)
        products = []
        for i in demand.index:
            for j in capacity.index:
                if _get_distance(i,j)<=service_type.accessibility*2:
                    products.append((i,j))
        x = LpVariable.dicts("Route", products, 0, None, cat=LpInteger)
        prob += lpSum(_get_weight(n, m) * x[n, m] for n, m in products)
        for n in demand.index:
            capacity_products = [tpl[1] for tpl in filter(lambda tpl : tpl[0] == n,products)]
            prob += lpSum(x[n, m] for m in capacity_products) <= demand.loc[n, DEMAND_LEFT_COLUMN]
        for m in capacity.index:
            demand_products = [tpl[0] for tpl in filter(lambda tpl : tpl[1] == m,products)]
            prob += lpSum(x[n, m] for n in demand_products) <= capacity.loc[m, CAPACITY_LEFT_COLUMN]
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
                # if a != fictive_demand and b != fictive_capacity:
                if distance <= service_type.accessibility:
                    gdf.loc[a, DEMAND_WITHIN_COLUMN] += value
                else:
                    gdf.loc[a, DEMAND_WITHOUT_COLUMN] += value
                gdf.loc[a, DEMAND_LEFT_COLUMN] -= value
                gdf.loc[b, CAPACITY_LEFT_COLUMN] -= value
                links.append({
                    'from': a,
                    'to': b,
                    'demand': value,
                    'within': distance <= service_type.accessibility
                })

        # if fictive_town_id is not None:
        #     gdf = gdf.drop(labels=[fictive_town_id])

        return gdf, links
