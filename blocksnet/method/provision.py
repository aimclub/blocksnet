"""
Service type provisions assessment module.
"""
from loguru import logger
import geopandas as gpd
import pandas as pd
from pydantic import Field
from pulp import PULP_CBC_CMD, LpMaximize, LpProblem, LpVariable, lpSum, LpInteger
from ..models import ServiceType
from .base_method import BaseMethod
from enum import Enum

PROVISION_COLUMN = "provision"

PLOT_KWARGS = {"column": PROVISION_COLUMN, "cmap": "RdYlGn", "vmin": 0, "vmax": 1, "legend": True}


class ProvisionMethod(Enum):
    """
    Enum for different methods of service provision assessment.

    Attributes
    ----------
    GREEDY : str
        Greedy method for service provision.
    GRAVITATIONAL : str
        Linear method for service provision, where distances are taken into account as squares.
    LINEAR : str
        Linear method for service provision, where distances are taken into account linearly.
    """

    GREEDY = "greedy"
    GRAVITATIONAL = "gravitational"
    LINEAR = "linear"


class Provision(BaseMethod):
    """
    Class for assessing provision of services to urban areas.

    This class provides various methods to evaluate the provision of services
    such as healthcare, education, etc., for a given city model. The provision
    can be assessed using multiple methods like gravitational models, greedy
    allocation, and linear programming-based transportation problem solutions.

    Methods
    -------
    plot(gdf: gpd.GeoDataFrame, linewidth: float = 0.1, figsize: tuple[int, int] = (10, 10)) -> None
        Visualizes provision assessment results for a given GeoDataFrame.

    stat(gdf: gpd.GeoDataFrame) -> dict[str, float]
        Computes basic statistics of provision (mean, median, min, max) from a GeoDataFrame.

    total(gdf: gpd.GeoDataFrame) -> float
        Calculates the total provision ratio from the given GeoDataFrame.

    get_bounds(service_type: ServiceType | str, update_df: pd.DataFrame = None) -> tuple[float, float]
        Returns the lower and upper bounds of provision for a given service type.

    calculate(service_type: ServiceType | str, update_df: pd.DataFrame | None = None, method: ProvisionMethod = ProvisionMethod.GRAVITATIONAL, self_supply: bool = False) -> gpd.GeoDataFrame
        Performs provision assessment for a specified service type and method.
    """

    max_depth: int = Field(gt=0, default=3)

    def plot(self, gdf: gpd.GeoDataFrame, linewidth: float = 0.1, figsize: tuple[int, int] = (10, 10)):
        """
        Visualizes provision assessment results for a given GeoDataFrame.

        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            GeoDataFrame containing spatial data with provision assessment results.
        linewidth : float
            Size of polygons' borders, by default 0.1.
        figsize : tuple of int, optional
            Size of the plot in inches, by default (10, 10).

        Returns
        -------
        None
        """
        ax = gdf.plot(color="#ddd", figsize=figsize)
        gdf.plot(ax=ax, linewidth=linewidth, **PLOT_KWARGS)
        ax.set_title("Provision: " + f"{self.total(gdf): .3f}")
        ax.set_axis_off()

    def _get_blocks_gdf(self, service_type: ServiceType, update_df: pd.DataFrame | None = None) -> gpd.GeoDataFrame:
        """
        Generates a GeoDataFrame of city blocks with updated service capacities and demands.

        Parameters
        ----------
        service_type : ServiceType
            The service type for which provision assessment is being calculated.
        update_df : pandas.DataFrame, optional
            DataFrame containing updates to population or capacity, by default None.

        Returns
        -------
        geopandas.GeoDataFrame
            GeoDataFrame containing the geometry, demand, and capacity for each block.
        """
        capacity_column = f"capacity_{service_type.name}"
        gdf = self.city_model.get_blocks_gdf()[["geometry", "population", capacity_column]].fillna(0)
        gdf = gdf.rename(columns={capacity_column: "capacity"})
        if update_df is not None:
            if "population" in update_df.columns:
                gdf["population"] = gdf["population"].add(update_df["population"].fillna(0), fill_value=0)
            if service_type.name in update_df.columns:
                gdf["capacity"] = gdf["capacity"].add(update_df[service_type.name].fillna(0), fill_value=0)
        gdf["population"] = gdf["population"].apply(service_type.calculate_in_need)
        gdf = gdf.rename(columns={"population": "demand"})
        gdf["capacity_left"] = gdf["capacity"]
        gdf["demand_left"] = gdf["demand"]
        gdf["demand_within"] = 0
        gdf["demand_without"] = 0
        return gdf

    @classmethod
    def stat(cls, gdf: gpd.GeoDataFrame) -> dict[str, float]:
        """
        Computes basic statistics of provision (mean, median, min, max) from a GeoDataFrame.

        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            GeoDataFrame containing provision data.

        Returns
        -------
        dict
            Dictionary with keys 'mean', 'median', 'min', and 'max' representing provision statistics.
        """
        return {
            "mean": gdf["provision"].mean(),
            "median": gdf["provision"].median(),
            "min": gdf["provision"].min(),
            "max": gdf["provision"].max(),
        }

    @classmethod
    def total(cls, gdf: gpd.GeoDataFrame) -> float:
        """
        Calculates the total provision by dividing the sum of met demand
        by the total demand for all blocks in the GeoDataFrame.

        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            GeoDataFrame containing the columns 'demand_within' and 'demand',
            representing the met demand and total demand for each block.

        Returns
        -------
        float
            The ratio of total met demand to total demand, representing overall provision.
        """
        return gdf["demand_within"].sum() / gdf["demand"].sum()

    @staticmethod
    def _get_lower_bound(gdf: gpd.GeoDataFrame) -> float:
        """
        Calculates the lower bound of provision, assuming each block
        meets its demand up to its capacity.

        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            GeoDataFrame containing 'capacity' and 'demand' columns.

        Returns
        -------
        float
            The lower bound of provision as the ratio of the total met demand
            (limited by capacity) to the total demand.
        """
        gdf = gdf.copy()
        gdf["demand_within"] = gdf.apply(lambda x: min(x["capacity"], x["demand"]), axis=1)
        return gdf["demand_within"].sum() / gdf["demand"].sum()

    @staticmethod
    def _get_upper_bound(gdf: gpd.GeoDataFrame) -> float:
        """
        Calculates the upper bound of provision, assuming total capacity can be
        fully allocated to meet total demand.

        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            GeoDataFrame containing 'capacity' and 'demand' columns.

        Returns
        -------
        float
            The upper bound of provision, calculated as the minimum of the total
            capacity-to-demand ratio or 1 (full provision).
        """
        gdf = gdf.copy()
        capacity = gdf["capacity"].sum()
        demand = gdf["demand"].sum()
        return min(capacity / demand, 1)

    def get_bounds(self, service_type: ServiceType | str, update_df: pd.DataFrame = None) -> tuple[float, float]:
        """
        Returns the lower and upper bounds of provision for a given service type.

        Parameters
        ----------
        service_type : ServiceType or str
            The service type for which provision bounds are calculated.
        update_df : pandas.DataFrame, optional
            DataFrame containing updates to population or capacity, by default None.

        Returns
        -------
        tuple of float
            A tuple (lower_bound, upper_bound) representing the lower and upper
            bounds of provision.
        """
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
        """
        Performs provision assessment for a specified service type and method.

        Parameters
        ----------
        service_type : ServiceType or str
            The type of service for which the provision is calculated.
        update_df : pandas.DataFrame, optional
            Updated DataFrame for blocks (demand or capacity changes), by default None.
        method : ProvisionMethod, optional
            The provision method to be used (GRAVITATIONAL by default).
        self_supply : bool, optional
            If True, blocks are allowed to meet their own demand directly using their
            own capacity, by default False.

        Returns
        -------
        geopandas.GeoDataFrame
            GeoDataFrame with provision data, including 'demand_within', 'demand_left',
            'capacity_left', and 'provision'.
        """
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
            gdf = self._lp_provision(gdf, service_type, method)

        gdf["provision"] = gdf["demand_within"] / gdf["demand"]

        if self.verbose:
            logger.success("Provision assessment finished")

        return gdf

    def _lp_provision(
        self,
        gdf: gpd.GeoDataFrame,
        service_type: ServiceType,
        method: ProvisionMethod,
        depth: int = 1,
    ) -> gpd.GeoDataFrame:
        """
        Solves the provision problem using a Linear Programming (LP) solver.
        Loops itself till capacity or demand left meet 0.

        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            GeoDataFrame containing blocks with demand and capacity.
        service_type : ServiceType
            The type of service for which provision is being calculated.
        method : ProvisionMethod
            The method used to calculate provision (LINEAR or GRAVITATIONAL).
        depth : int
            Current depth in the loop, default 1.

        Returns
        -------
        geopandas.GeoDataFrame
            Updated GeoDataFrame with provision results, including updates to
            'demand_within', 'demand_without', 'demand_left', and 'capacity_left'.
        """

        selection_range = depth * service_type.accessibility

        demand = gdf.loc[gdf["demand_left"] > 0]
        capacity = gdf.loc[gdf["capacity_left"] > 0]

        def _get_distance(id1: int, id2: int):
            distance = self.city_model.accessibility_matrix.loc[id1, id2]
            return distance if distance > 1 else 1

        def _get_weight(id1: int, id2: int):
            distance = _get_distance(id1, id2)
            if method == ProvisionMethod.LINEAR:
                return 1 / distance
            return demand.loc[id1, "demand_left"] / (distance * distance)  # * capacity.loc[id2, 'capacity_left']

        if self.verbose:
            logger.info(f"Setting an LP problem for accessibility = {selection_range} : {len(demand)}x{len(capacity)}")

        prob = LpProblem("Provision", LpMaximize)
        # Precompute distance and filter products
        products = [
            (i, j)
            for i in demand.index
            for j in capacity.index
            if _get_distance(i, j) <= selection_range  # service_type.accessibility * 2
        ]

        # Create the decision variable dictionary
        x = LpVariable.dicts("Route", products, 0, None, cat=LpInteger)  #

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
            # value = round(value)
            if value > 0:
                if distance <= service_type.accessibility:
                    gdf.loc[a, "demand_within"] += value
                else:
                    gdf.loc[a, "demand_without"] += value
                gdf.loc[a, "demand_left"] -= value
                gdf.loc[b, "capacity_left"] -= value

        if gdf["demand_left"].sum() > 0 and gdf["capacity_left"].sum() > 0 and depth < self.max_depth:
            return self._lp_provision(gdf, service_type, method, depth + 1)
        return gdf

    def _greedy_provision(self, gdf: gpd.GeoDataFrame, service_type: ServiceType) -> gpd.GeoDataFrame:
        """
        Iteratively assigns demand to the closest available capacity using a greedy method.

        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            GeoDataFrame containing blocks with demand and capacity.
        service_type : ServiceType
            The type of service for which provision is being calculated.

        Returns
        -------
        geopandas.GeoDataFrame
            Updated GeoDataFrame with provision results, including updates to
            'demand_within', 'demand_without', 'demand_left', and 'capacity_left'.
        """
        demand_gdf = gdf.loc[gdf["demand_left"] > 0]
        capacity_gdf = gdf.loc[gdf["capacity_left"] > 0]

        demand_blocks = [self.city_model[i] for i in demand_gdf.index]
        capacity_blocks = [self.city_model[i] for i in capacity_gdf.index]

        while gdf["demand_left"].sum() > 0 and gdf["capacity_left"].sum() > 0:

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
