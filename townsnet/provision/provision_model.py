import math
import geopandas as gpd
import pandas as pd
import pandera as pa
from loguru import logger
from pandera.typing import Index, Series
from pulp import PULP_CBC_CMD, LpMaximize, LpProblem, LpVariable, lpSum, LpInteger
from shapely import Point
from ..base_schema import BaseSchema
from .service_type import ServiceType

POPULATION_COLUMN = 'population'
SUPPLY_COLUMN = 'supply'

CAPACITY_COLUMN = 'capacity'
CAPACITY_LEFT_COLUMN = 'capacity_left'
DEMAND_COLUMN = 'demand'
DEMAND_LEFT_COLUMN = 'demand_left'
DEMAND_WITHIN_COLUMN = 'demand_within'
DEMAND_WITHOUT_COLUMN = 'demand_without'
PROVISION_COLUMN = "provision"

class SuppliesSchema(pa.DataFrameModel):
  idx : Index[int] = pa.Field(unique=True)
  supply : Series[float] = pa.Field(coerce = True)
  # delta : Series[float] = pa.Field(coerce = True, nullable=True)

  # class Config:
  #   add_missing_columns = True

class TownsSchema(BaseSchema):
  idx : Index[int] = pa.Field(unique=True)
  population : Series[int] = pa.Field(coerce = True)
  _geom_types = [Point]
  # delta : Series[float] = pa.Field(coerce = True, nullable=True)

  @pa.parser('geometry')
  @classmethod
  def parse_geometry(cls, series):
    name = series.name
    return series.representative_point().rename(name)

  # class Config:
  #   add_missing_columns = True

class ProvisionModel():

    def __init__(self, towns_gdf : gpd.GeoDataFrame, acc_mx : pd.DataFrame, max_depth : int = 5, verbose : bool = True):
        assert (towns_gdf.index == acc_mx.index).all(), 'Towns index and acc_mx index should match'
        assert (towns_gdf.index == acc_mx.columns).all(), 'Towns index and acc_mx columns should match'
        self.towns_gdf = TownsSchema(towns_gdf)
        self.accessibility_matrix = acc_mx.copy()
        self.max_depth = max_depth
        self.verbose = verbose

    def _preprocess_gdf(self, supplies_df : SuppliesSchema, service_type : ServiceType):
        gdf = self.towns_gdf.copy()
        gdf[DEMAND_COLUMN] = gdf[POPULATION_COLUMN].apply(lambda p : math.ceil(p/1000*service_type.supply_value))
        gdf[CAPACITY_COLUMN] = supplies_df[SUPPLY_COLUMN]
        gdf[CAPACITY_LEFT_COLUMN] = gdf[CAPACITY_COLUMN]
        gdf[DEMAND_LEFT_COLUMN] = gdf[DEMAND_COLUMN]
        gdf[DEMAND_WITHIN_COLUMN] = 0
        gdf[DEMAND_WITHOUT_COLUMN] = 0
        return gdf
    
    @classmethod
    def total(cls, gdf: gpd.GeoDataFrame) -> float:
        return gdf[DEMAND_WITHIN_COLUMN].sum() / gdf[DEMAND_COLUMN].sum()
    
    @staticmethod
    def agregate(towns : gpd.GeoDataFrame, units : gpd.GeoDataFrame):
        units = units[['geometry']].copy()
        sjoin = units.sjoin(towns, how='left')
        sjoin = sjoin.reset_index()
        sjoin = sjoin.groupby('territory_id').agg({
            CAPACITY_COLUMN : 'sum',
            CAPACITY_LEFT_COLUMN : 'sum',
            DEMAND_COLUMN : 'sum',
            DEMAND_LEFT_COLUMN : 'sum',
            DEMAND_WITHIN_COLUMN : 'sum',
            DEMAND_WITHOUT_COLUMN : 'sum'
        })
        sjoin[PROVISION_COLUMN] = sjoin[DEMAND_WITHIN_COLUMN] / sjoin[DEMAND_COLUMN]
        return units.merge(sjoin, left_index=True, right_index=True)

    def calculate(
        self, supplies_df : pd.DataFrame, service_type: ServiceType, self_supply: bool = True,
    ) -> gpd.GeoDataFrame:
        
        gdf = self._preprocess_gdf(supplies_df, service_type)

        if self_supply:
            supply : pd.Series = gdf.apply(lambda x: min(x[DEMAND_COLUMN], x[CAPACITY_COLUMN]), axis=1)
            gdf[DEMAND_WITHIN_COLUMN] += supply
            gdf[DEMAND_LEFT_COLUMN] -= supply
            gdf[CAPACITY_LEFT_COLUMN] -= supply

        gdf = self._lp_provision(gdf, service_type)

        gdf[PROVISION_COLUMN] = gdf[DEMAND_WITHIN_COLUMN] / gdf[DEMAND_COLUMN]

        if self.verbose:
            logger.success("Provision assessment finished")

        return gdf

    def _lp_provision(
        self,
        gdf: gpd.GeoDataFrame,
        service_type: ServiceType,
        depth: int = 1,
    ) -> gpd.GeoDataFrame:
        
        selection_range = depth * service_type.accessibility_value

        def _get_distance(id1: int, id2: int):
            distance = self.accessibility_matrix.loc[id1, id2]
            return distance if distance > 1 else 1

        def _get_weight(id1: int, id2: int):
            distance = _get_distance(id1, id2)
            return 1 / (distance * distance)

        demand = gdf.loc[gdf[DEMAND_LEFT_COLUMN] > 0]
        capacity = gdf.loc[gdf[CAPACITY_LEFT_COLUMN] > 0]

        if self.verbose:
            logger.info(f"Setting an LP problem for depth = {depth} : {len(demand)}x{len(capacity)}")

        prob = LpProblem("Provision", LpMaximize)
        # Precompute distance and filter products
        products = [
            (i, j)
            for i in demand.index
            for j in capacity.index
            if _get_distance(i, j) <= selection_range  # service_type.accessibility * 2
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
            prob += lpSum(demand_constraints[n]) <= demand.loc[n, DEMAND_LEFT_COLUMN]

        # Add Capacity Constraints
        for m in capacity.index:
            prob += lpSum(capacity_constraints[m]) <= capacity.loc[m, CAPACITY_LEFT_COLUMN]

        # if self.verbose:
        #     logger.info("Solving the problem")
        prob.solve(PULP_CBC_CMD(msg=False))

        # if self.verbose:
        #     logger.info("Restoring values from variables")

        for var in prob.variables():
            value = var.value()
            name = var.name.replace("(", "").replace(")", "").replace(",", "").split("_")
            if name[2] == "dummy":
                continue
            a = int(name[1])
            b = int(name[2])
            distance = _get_distance(a, b)
            if value > 0:
                if distance <= service_type.accessibility_value:
                    gdf.loc[a, DEMAND_WITHIN_COLUMN] += value
                else:
                    gdf.loc[a, DEMAND_WITHOUT_COLUMN] += value
                gdf.loc[a, DEMAND_LEFT_COLUMN] -= value
                gdf.loc[b, CAPACITY_LEFT_COLUMN] -= value

        if gdf[DEMAND_LEFT_COLUMN].sum() > 0 and gdf[CAPACITY_LEFT_COLUMN].sum() > 0 and depth < self.max_depth:
            return self._lp_provision(gdf, service_type, depth + 1)
        return gdf