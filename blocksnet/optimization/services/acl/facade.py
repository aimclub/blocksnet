import copy
import csv
from functools import reduce
from math import floor as round_floor
from operator import attrgetter
from typing import Dict, List, Set

import geopandas as gpd
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from blocksnet.config import service_types_config
from blocksnet.config.land_use.common.gsi_ranges import gsi_ranges
from blocksnet.enums import LandUse
from blocksnet.optimization.services.acl.checkers import AreaChecker, CapacityChecker
from blocksnet.optimization.services.acl.variable_adapters import AreaSolution, BlockSolution, VariableAdapter
from blocksnet.optimization.services.common import ServicesContainer
from blocksnet.optimization.services.common.variable import Variable
from blocksnet.optimization.services.schemas import ServicesSchema
from blocksnet.relations import validate_accessibility_matrix

from .provision_adapter import BFA_COEF, LIVING_DEMAND, ProvisionAdapter


class Facade:
    """
    Facade class providing a simplified interface for urban block service optimization.

    This class handles the initialization of service optimization components and provides
    methods to check constraints, calculate provisions, and manage service variables.
    """

    def __init__(
        self,
        var_adapter: VariableAdapter,
        accessibility_matrix: pd.DataFrame,
        blocks_df: gpd.GeoDataFrame,
        blocks_lu: Dict[int, LandUse],
    ) -> None:
        """
        Initialize the Facade with required data structures.

        Parameters
        ----------
        var_adapter : VariableAdapter
            Adapter for converting between solution vectors and service variables.
        accessibility_matrix : pd.DataFrame
            Matrix representing accessibility between blocks and services.
        blocks_df : gpd.GeoDataFrame
            GeoDataFrame containing block geometries and attributes.
        blocks_lu : Dict[int, LandUse]
            Dictionary mapping block IDs to land use types.
        """
        validate_accessibility_matrix(accessibility_matrix, blocks_df)
        self._blocks_lu: Dict[int, LandUse] = blocks_lu
        self._area_checker: AreaChecker = AreaChecker(blocks_lu, blocks_df)

        # Determine valid service types based on land use
        blocks_service_types: Set[str] = reduce(
            lambda x, y: x | y, [set(service_types_config[lu]) for lu in blocks_lu.values()]
        )
        self._blocks_service_types: Set[str] = set(st for st in blocks_service_types)
        self._chosen_service_types: Set[str] = set()

        # Initialize provision adapter
        self._provision_adapter: ProvisionAdapter = ProvisionAdapter(blocks_lu, accessibility_matrix, blocks_df)

        # Initial provisions
        self._start_provisions: Dict[str, float] = {}

        # Initialize solution converter and checkers
        self._converter = var_adapter
        self._converter.update_bfa(self._area_checker.build_floor_areas)
        self._capacity_checker: CapacityChecker = CapacityChecker(
            list(blocks_lu.keys()),
            self._provision_adapter._accessibility_matrix,
        )
        self._converter.update_cap(self._capacity_checker._demands)
        self.num_params = 0
        self._last_provisions: Dict[str, float] = {}
        self._last_blocks_services_demand: Dict[int, Dict[str, int]] | None = None

    @property
    def start_provisions(self) -> Dict[str, float]:
        """
        Get the initial provision values for all service types.

        Returns
        -------
        Dict[str, float]
            Dictionary mapping service types to their initial provision values.
        """
        return self._start_provisions

    @property
    def last_provisions(self) -> Dict[str, float]:
        """
        Get the last trial's provision values for all service types.

        Returns
        -------
        Dict[str, float]
            Dictionary mapping service types to their provision values from the last solution.
        """
        return self._last_provisions

    def solution_to_services_df(self, solution: Dict[str, int]) -> pd.DataFrame:
        """
        Convert a solution dictionary to a DataFrame of service allocations.

        Parameters
        ----------
        solution : Dict[str, int]
            Dictionary mapping variable names to their values in the solution.

        Returns
        -------
        pd.DataFrame
            DataFrame containing service allocation details with columns:
            - block_id: ID of the block
            - service_type: Type of service
            - site_area: Area occupied by the service
            - build_floor_area: Floor area of the service
            - capacity: Capacity of the service
            - count: Number of service units allocated
        """
        x = np.zeros(self.num_params)
        for var_name, var_val in solution.items():
            x[int(var_name[2:])] = var_val
        X = self._converter(x)

        # Save placed services
        xs = [
            {
                "block_id": x.block_id,
                "service_type": x.service_type,
                "site_area": x.site_area,
                "build_floor_area": x.build_floor_area,
                "capacity": x.capacity,
                "count": x.count,
            }
            for x in X
        ]
        df = pd.DataFrame(list(xs))
        if len(xs) == 0:
            return df
        return df[df["count"] != 0]

    def add_service_type(self, name: str, weight: float, services_df: pd.DataFrame) -> None:
        """
        Add a service type to the optimization problem.

        Parameters
        ----------
        name : str
            Name of the service type to add.
        weight : float
            Weight factor for this service type in the optimization.
        services_df : pd.DataFrame
            DataFrame containing service specifications and parameters.
        """
        if name not in self._blocks_service_types:
            return

        services_df = ServicesSchema(services_df)
        services_container = ServicesContainer(name=name, weight=weight, services_df=services_df)
        self._provision_adapter.add_service_type(services_container)

        provision_df = self._provision_adapter.get_start_provision_df(name)

        if provision_df is None:
            return

        prov = self._provision_adapter.calculate_provision(name)

        self._capacity_checker.add_service_type(name, self._provision_adapter.get_start_provision_df(name))
        self._converter.add_service_type_vars(name)
        self.num_params = len(self._converter)
        self._chosen_service_types.add(name)
        self._start_provisions[name] = prov
        self._last_provisions[name] = prov

    def check_constraints(self, x: ArrayLike) -> bool:
        """
        Check if a solution satisfies all constraints.

        Parameters
        ----------
        x : ArrayLike
            Solution vector to validate.

        Returns
        -------
        bool
            True if all constraints are satisfied, False otherwise.
        """
        X = self._converter(x)
        return self._area_checker.check_constraints(X) and self._capacity_checker.check_constraints(
            X, self._last_blocks_services_demand
        )

    def get_X(self, x: ArrayLike) -> List[Variable]:
        """
        Get the list of Variable objects corresponding to a solution vector.

        Parameters
        ----------
        x : ArrayLike
            Solution vector to convert.

        Returns
        -------
        List[Variable]
            List of Variable objects representing the solution.
        """
        self._converter(x)
        return copy.deepcopy(self._converter.X)

    def get_lower_bound_var_val(self, var_num: int) -> float:
        """
        Get the lower bound value for a specific variable.

        Parameters
        ----------
        var_num : int
            Index of the variable to query.

        Returns
        -------
        float
            Lower bound value for the specified variable.
        """
        var = self._converter.X[var_num]
        if isinstance(self._converter, AreaSolution):
            limits = self.get_limits_var(var_num)
            ub = self.get_upper_bound_var(var_num)
            units = [
                unit
                for unit in self._converter._units_lists[var.service_type]
                if unit.site_area <= ub and unit.build_floor_area <= limits[1] and unit.capacity <= limits[2]
            ]
            if len(units) == 0:
                return 0
            return units[0].site_area
        return 1

    def get_max_capacity(self, block_id: int, service: str) -> float:
        """
        Get the maximum capacity for a service type in a block.

        Parameters
        ----------
        block_id : int
            ID of the block to query.
        service : str
            Service type identifier.

        Returns
        -------
        float
            Maximum capacity for the specified service in the given block.
        """
        demand = 0
        if self._last_blocks_services_demand is not None:
            demand = self._last_blocks_services_demand[block_id][service]
        return self._capacity_checker.get_demand(block_id, service, demand)

    def get_max_capacities(self, block_id: int) -> Dict[str, float]:
        """
        Get maximum capacities for all valid services in a block.

        Parameters
        ----------
        block_id : int
            ID of the block to query.

        Returns
        -------
        Dict[str, float]
            Dictionary mapping service types to their maximum capacities.
        """
        services = self.get_block_services(block_id)
        return {st: self.get_max_capacity(block_id, st) for st in services}

    def get_distance(self, x: ArrayLike) -> float:
        """
        Calculate the total distance metric for a solution.

        Parameters
        ----------
        x : ArrayLike
            Solution vector to evaluate.

        Returns
        -------
        float
            Total distance metric for the solution.
        """
        X = self._converter(x)
        return self._area_checker.get_distance_for_entire_solution(X)

    def get_max_distance(self) -> float:
        """
        Calculate the maximum possible distance metric.

        Returns
        -------
        float
            Maximum possible distance metric when all variables are at maximum capacity.
        """
        X = copy.deepcopy(self._converter.X)
        for i, x in enumerate(X):
            X[i].count = self.get_upper_bound_var(i)
        return self._area_checker.get_distance_for_entire_solution(X)

    def get_block_services(self, block_id: int) -> Set[str]:
        """
        Get valid service types for a block based on land use.

        Parameters
        ----------
        block_id : int
            ID of the block to query.

        Returns
        -------
        Set[str]
            Set of valid service types for the block.
        """
        chosen_service_types = set(st for st in set(service_types_config[self._blocks_lu[block_id]]))
        return chosen_service_types & self._chosen_service_types

    def get_changed_services(self, x_last: ArrayLike, x: ArrayLike) -> Set[str]:
        """
        Identify services that changed between two solutions.

        Parameters
        ----------
        x_last : ArrayLike
            Previous solution vector.
        x : ArrayLike
            Current solution vector.

        Returns
        -------
        Set[str]
            Set of service types that changed between solutions.
        """
        services_vars = {st: [] for st in self._chosen_service_types}
        services_vars_last = {st: [] for st in self._chosen_service_types}
        for var_last in self._converter(x_last):
            services_vars_last[var_last.service_type].append(copy.deepcopy(var_last))
        for var in self._converter(x):
            services_vars[var.service_type].append(copy.deepcopy(var))

        changed_services = self._chosen_service_types.copy()
        for st in self._chosen_service_types:
            changed = False
            services_vars[st].sort(key=attrgetter("block_id", "capacity", "site_area", "build_floor_area"))
            services_vars_last[st].sort(key=attrgetter("block_id", "capacity", "site_area", "build_floor_area"))
            for var, var_last in zip(services_vars[st], services_vars_last[st]):
                if (
                    var.capacity != var_last.capacity
                    or var.site_area != var_last.site_area
                    or var.build_floor_area != var_last.build_floor_area
                    or var.count > 0
                    and var.count != var_last.count
                ):
                    changed = True
                    break
            if not changed:
                changed_services.remove(st)

        return changed_services

    def get_provisions(self, x_last: ArrayLike, x: ArrayLike) -> tuple[Dict[str, float], Set[str]]:
        """
        Calculate provisions for changed services between solutions.

        Parameters
        ----------
        x_last : ArrayLike
            Previous solution vector.
        x : ArrayLike
            Current solution vector.

        Returns
        -------
        tuple[Dict[str, float], Set[str]]
            Tuple containing:
            - Dictionary mapping changed service types to their provision values
            - Set of changed service types
        """
        changed_services = self.get_changed_services(x_last, x)
        if self._start_provisions == self._last_provisions:
            changed_services = copy.deepcopy(self._chosen_service_types)

        X = self._converter(x)
        vars_df = self._converter._variables_to_df(X)
        blocks_services_demand = self.get_delta_demand(x)
        for _, sd in blocks_services_demand.items():
            for service_type, demand in sd.items():
                if (
                    demand > 0
                    and service_type not in changed_services
                    and abs(self._start_provisions[service_type] - 1.0) < 1e-8
                ):  # force recalculate provisions for populated services
                    changed_services.add(service_type)
        provisions = {
            st: self._provision_adapter.calculate_provision(st, self._area_checker.build_floor_areas, vars_df)
            for st in changed_services
        }

        fixed_services = self._chosen_service_types - changed_services

        for st in fixed_services:
            if sum(var.count for var in X if var.service_type == st) == 0:
                provisions[st] = self._start_provisions[st]
            else:
                provisions[st] = self._last_provisions[st]

        self._last_provisions = provisions
        if self._last_blocks_services_demand is not None:
            self._last_blocks_services_demand = None
        else:
            self._last_blocks_services_demand = self.get_delta_demand(x)
        self._converter.update_cap({block_id: self.get_max_capacities(block_id) for block_id in self._blocks_lu.keys()})

        return provisions, changed_services

    def get_all_provisions(self, x: ArrayLike) -> tuple[Dict[str, float], Set[str]]:
        """
        Calculate provisions for all services in a solution.

        Parameters
        ----------
        x : ArrayLike
            Solution vector to evaluate.

        Returns
        -------
        tuple[Dict[str, float], Set[str]]
            Tuple containing:
            - Dictionary mapping all service types to their provision values
            - Set of all service types
        """
        X = self._converter(x)
        vars_df = self._converter._variables_to_df(X)
        provisions = {
            st: self._provision_adapter.calculate_provision(st, self._area_checker.build_floor_areas, vars_df)
            for st in self._chosen_service_types
        }

        if np.count_nonzero(x) != 0:  # first trial (not null)
            self._last_provisions = provisions

        return provisions, self._chosen_service_types

    def get_upper_bound_var(self, var_num: int) -> int:
        """
        Get the upper bound for a variable.

        Parameters
        ----------
        var_num : int
            Variable index.

        Returns
        -------
        int
            Upper bound value considering both area and demand constraints.
        """
        var = self._converter.X[var_num]
        area_ub = self._area_checker.get_ub_var(var)
        demand = 0
        if self._last_blocks_services_demand is not None:
            demand = self._last_blocks_services_demand[var.block_id][var.service_type]
        demand_ub = self._capacity_checker.get_ub_var(var, demand)
        if demand_ub == -1:
            if area_ub == -1:
                return 1e9
            return area_ub
        if area_ub == -1:
            return demand_ub
        return min(demand_ub, area_ub)

    def get_total_population(self, solution: Dict[int, Dict[str, int]]) -> Dict[int, int]:
        """
        Calculate total population including base population and solution additions.

        Parameters
        ----------
        solution : Dict[int, Dict[str, int]]
            Solution dictionary mapping blocks to service populations.

        Returns
        -------
        Dict[int, int]
            Dictionary mapping block IDs to their total population.
        """
        x = np.zeros(self.num_params)
        for var_name, var_val in solution.items():
            x[int(var_name[2:])] = var_val
        blocks_population = self.get_delta_population(x)
        blocks_population_total: Dict[int, int] = {}
        for block, population in blocks_population.items():
            if block not in blocks_population_total.keys():
                blocks_population_total.update({block: 0})
            blocks_population_total[block] = population
            blocks_population_total[block] += self._provision_adapter._blocks_df.loc[block, "population"]
        return blocks_population_total

    def get_delta_demand(self, x: ArrayLike) -> Dict[int, Dict[str, int]]:
        """
        Calculate additional demand generated by a solution.

        Parameters
        ----------
        x : ArrayLike
            Solution vector to evaluate.

        Returns
        -------
        Dict[int, Dict[str, int]]
            Nested dictionary mapping block IDs to service types to their additional demand.
        """
        blocks_demand: Dict[int, Dict[str, int]] = {
            block_id: {service: 0 for service in self.get_block_services(block_id)}
            for block_id in self._blocks_lu.keys()
        }
        X = self._converter(x)
        vars_df = self._converter._variables_to_df(X)
        agg_total_build_area = vars_df.groupby("block_id").agg(
            {"total_build_floor_area": "sum"}
        )  # BEFORE cutting variables_df
        for service_type in self._chosen_service_types:
            variables_df = vars_df[vars_df.service_type == service_type]

            # Aggregate capacity updates by block
            delta_df = variables_df.groupby("block_id").agg({"total_capacity": "sum"})

            # Get service type parameters
            _, demand, _ = service_types_config[service_type].values()

            delta_df["max_population"] = 0

            # 3' BFA Refill
            for block_id in self._blocks_lu.keys():
                block_id = int(block_id)
                if self._blocks_lu[block_id].name == "RESIDENTIAL":
                    bfa_unit = (
                        self._area_checker.build_floor_areas[block_id]
                        - agg_total_build_area.loc[block_id, "total_build_floor_area"]
                    )
                    if block_id not in blocks_demand.keys():
                        blocks_demand.update({block_id: {}})
                    if service_type not in blocks_demand[block_id].keys():
                        blocks_demand[block_id].update({service_type: 0})
                    blocks_demand[block_id][service_type] = round_floor(
                        (bfa_unit * (1 / BFA_COEF - 1) * demand) / (LIVING_DEMAND * 1000)
                    )
                    if blocks_demand[block_id][service_type] < 0:
                        raise ValueError(
                            f"Negative population for block {block_id} and service {service_type}. BFA unit: {bfa_unit}, demand: {demand}"
                        )
        return blocks_demand

    def get_delta_population(self, x: ArrayLike) -> Dict[int, int]:
        """
        Calculate population changes resulting from a solution.

        Parameters
        ----------
        x : ArrayLike
            Solution vector to evaluate.

        Returns
        -------
        Dict[int, int]
            Dictionary mapping block IDs to their population change.
        """
        blocks_population: Dict[int, int] = {block_id: 0 for block_id in self._blocks_lu.keys()}
        X = self._converter(x)
        vars_df = self._converter._variables_to_df(X)
        agg_total_build_area = vars_df.groupby("block_id").agg({"total_build_floor_area": "sum"})
        for block_id in self._blocks_lu.keys():
            if self._blocks_lu[block_id].name == "RESIDENTIAL":
                bfa_unit = (
                    self._area_checker.build_floor_areas[block_id]
                    - agg_total_build_area.loc[block_id, "total_build_floor_area"]
                )
                blocks_population[block_id] = round_floor((bfa_unit * (1 / BFA_COEF - 1)) / (LIVING_DEMAND))

        return blocks_population

    def save_delta_demand(self, solution: Dict[str, int]):
        """
        Save the demand changes from a solution to a CSV file.

        Parameters
        ----------
        solution : Dict[int, Dict[str, int]]
            Solution dictionary mapping blocks to service populations.
        """
        x = np.zeros(self.num_params)
        for var_name, var_val in solution.items():
            x[int(var_name[2:])] = var_val
        blocks_demand = self.get_delta_demand(x)
        with open("demand_services.csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            columns = ["Block id", "Service type", "Demand"]
            writer.writerow(columns)  # Write header row

            for block_id, sp in blocks_demand.items():
                for service_type, demand in sp.items():
                    writer.writerow([block_id, service_type, demand])

    def get_var_weights(self, var_num: int) -> ArrayLike:
        """
        Get the weight factors for a variable.

        Parameters
        ----------
        var_num : int
            Variable index.

        Returns
        -------
        ArrayLike
            Array containing weight factors for site area, build floor area, and capacity.
        """
        var = self._converter.X[var_num]
        return np.array([var.site_area, var.build_floor_area, var.capacity])

    def get_limits_var(self, var_num: int) -> ArrayLike:
        """
        Get the limit values for a variable's constraints.

        Parameters
        ----------
        var_num : int
            Variable index.

        Returns
        -------
        ArrayLike
            Array containing limit values for site area, build floor area, and capacity.
        """
        var = self._converter.X[var_num]
        site_area_limit = self._area_checker.site_areas[var.block_id]
        build_floor_area_limit = self._area_checker.build_floor_areas[var.block_id]
        capacity_limit = self.get_max_capacity(var.block_id, var.service_type)
        return np.array([site_area_limit, build_floor_area_limit, capacity_limit])

    def get_var_block_id(self, var_num: int) -> int:
        """
        Get the block ID associated with a variable.

        Parameters
        ----------
        var_num : int
            Variable index.

        Returns
        -------
        int
            Block ID associated with the variable.
        """
        return self._converter.X[var_num].block_id

    def get_var_land_use(self, var_num: int) -> LandUse:
        """
        Get the land use type for a variable's block.

        Parameters
        ----------
        var_num : int
            Variable index.

        Returns
        -------
        LandUse
            Land use type of the block associated with the variable.
        """
        return self._blocks_lu[self._converter.X[var_num].block_id]

    def get_service_name(self, var_num: int) -> str:
        """
        Get the service type name for a variable.

        Parameters
        ----------
        var_num : int
            Variable index.

        Returns
        -------
        str
            Service type name associated with the variable.
        """
        return self._converter.X[var_num].service_type

    def get_solution_area_df(self, solution: Dict[str, int]) -> pd.DataFrame:
        """
        Generate a DataFrame summarizing area statistics by quarter based on a solution.

        This method calculates key metrics including population, building floor area,
        living area, and footprint area for each quarter in the urban area.

        Parameters
        ----------
        solution : Dict[str, int]
            Dictionary representing the solution, where keys are variable names (format 'x_N')
            and values are the variable values (counts of service units).

        Returns
        -------
        pd.DataFrame
            DataFrame with quarter IDs as index and columns:
            - population: Total population in the quarter
            - build_floor_area: Total building floor area in the quarter
            - living_area: Residential living area in the quarter
            - footprint_area: Minimum footprint area based on land use GSI requirements
        """
        x = np.zeros(self.num_params)
        for var_name, var_val in solution.items():
            x[int(var_name[2:])] = var_val
        X = self._converter(x)

        blocks_population = self.get_total_population(solution)
        bfa_solution = {block_id: 0 for block_id in self._blocks_lu.keys()}
        for var in X:
            bfa_solution[var.block_id] += var.total_build_floor_area
        quarter_data = {}
        blocks_df = self._provision_adapter._blocks_df

        for quarter_id, block_ids in blocks_df.groupby(blocks_df.index).groups.items():
            quarter_stats = {"population": 0, "build_floor_area": 0, "living_area": 0, "footprint_area": 0}

            for block_id in block_ids:
                quarter_stats["population"] += blocks_population.get(block_id, 0)

                quarter_stats["build_floor_area"] += self._area_checker.build_floor_areas.get(block_id, 0)

                if self._blocks_lu.get(block_id) == LandUse.RESIDENTIAL:
                    quarter_stats["living_area"] += self._area_checker.build_floor_areas.get(
                        block_id, 0
                    ) - bfa_solution.get(block_id, 0)

                land_use = self._blocks_lu.get(block_id)
                min_gsi = gsi_ranges.get(land_use, (0, 0))[0]
                quarter_stats["footprint_area"] += self._area_checker.site_areas.get(block_id, 0) * min_gsi

            quarter_data[quarter_id] = quarter_stats

        result_df = pd.DataFrame.from_dict(quarter_data, orient="index")

        return result_df
