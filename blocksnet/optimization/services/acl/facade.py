import copy
from functools import reduce
from typing import Dict, List, Set

import geopandas as gpd
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from blocksnet.config import service_types_config
from blocksnet.enums import LandUse
from blocksnet.optimization.services.acl.checkers import AreaChecker, CapacityChecker
from blocksnet.optimization.services.acl.variable_adapters import VariableAdapter
from blocksnet.optimization.services.common import ServicesContainer
from blocksnet.optimization.services.schemas import ServicesSchema
from blocksnet.utils.validation import validate_matrix

from .provision_adapter import ProvisionAdapter


class Facade:
    """
    Facade class providing a simplified interface for urban block service optimization.

    This class handles the initialization of service optimization components and provides
    methods to check constraints, calculate provisions, and manage service variables.
    """

    def __init__(
        self,
        accessibility_matrix: pd.DataFrame,
        blocks_df: gpd.GeoDataFrame,
        blocks_lu: Dict[int, LandUse],
        var_adapter: VariableAdapter,
    ) -> None:
        """
        Initialize the Facade with required data structures.

        Parameters
        ----------
        accessibility_matrix : pd.DataFrame
            Matrix representing accessibility between blocks and services.
        blocks_df : gpd.GeoDataFrame
            GeoDataFrame containing block geometries and attributes.
        blocks_lu : Dict[int, LandUse]
            Dictionary mapping block IDs to land use types.
        var_adapter : VariableAdapter
            Adapter for converting between solution vectors and service variables.
        """
        validate_matrix(accessibility_matrix, blocks_df)
        self._blocks_lu: Dict[int, LandUse] = blocks_lu
        self._area_checker: AreaChecker = AreaChecker(blocks_df)

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
        self._capacity_checker: CapacityChecker = CapacityChecker(
            list(blocks_lu.keys()),
            self._provision_adapter._accessibility_matrix,
        )
        self.num_params = 0
        self._init_provisions: Dict[str, float] = {st: 0 for st in self._blocks_service_types}

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

        provision_df = self._provision_adapter.get_provision_df(name)

        if provision_df is None:
            return

        prov = self._provision_adapter.calculate_provision(name)

        if abs(prov - 1.0) > 1e-10:
            self._capacity_checker.add_service_type(name, self._provision_adapter.get_provision_df(name))
            self._converter.add_service_type_vars(name)
            self.num_params = len(self._converter)
            self._chosen_service_types.add(name)
            self._start_provisions[name] = prov

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
        return self._area_checker.check_constraints(X)

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
        return self._capacity_checker.get_demand(
            block_id,
            service,
            self._provision_adapter._accessibility_matrix,
            self._provision_adapter.get_provision_df(service),
        )

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

    def update_area(self, x: ArrayLike) -> None:
        """
        Update area allocations based on a solution.

        Parameters
        ----------
        x : ArrayLike
            Solution vector containing updated area allocations.
        """
        X = self._converter(x)
        self._area_checker.update_area(X)

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

    def get_changed_services(self, x_init: ArrayLike, x: ArrayLike) -> Set[str]:
        """
        Identify services that changed between two solutions.

        Parameters
        ----------
        x_init : ArrayLike
            Initial solution vector.
        x : ArrayLike
            Current solution vector.

        Returns
        -------
        Set[str]
            Set of service types that changed between solutions.
        """
        X_diff = self._converter(x - x_init)
        services_vars_diff = {
            st: np.array([var.count for var in X_diff if var.service_type == st]) for st in self._chosen_service_types
        }

        X = self._converter(x)
        services_vars = {
            st: np.array([var.count for var in X if var.service_type == st]) for st in self._chosen_service_types
        }

        changed_services = self._chosen_service_types.copy()
        for st in services_vars.keys():
            if np.count_nonzero(services_vars[st]) == 0 or np.count_nonzero(services_vars_diff[st]) == 0:
                changed_services.discard(st)

        return changed_services

    def get_provisions(self, x_init: ArrayLike, x: ArrayLike) -> Dict[str, float]:
        """
        Calculate provisions for changed services between solutions.

        Parameters
        ----------
        x_init : ArrayLike
            Initial solution vector.
        x : ArrayLike
            Current solution vector.

        Returns
        -------
        Dict[str, float]
            Dictionary mapping changed service types to their provision values.
        """
        X = self._converter(x)
        changed_services = self.get_changed_services(x_init, x)

        vars_df = self._converter._variables_to_df(X)
        provisions = {st: self._provision_adapter.calculate_provision(st, vars_df) for st in changed_services}

        init_services = self._init_provisions.keys() - changed_services
        for st in init_services:
            provisions[st] = self._init_provisions[st]

        return provisions

    def get_all_provisions(self, x: ArrayLike) -> Dict[str, float]:
        """
        Calculate provisions for all services in a solution.

        Parameters
        ----------
        x : ArrayLike
            Solution vector to evaluate.

        Returns
        -------
        Dict[str, float]
            Dictionary mapping all service types to their provision values.
        """
        X = self._converter(x)
        vars_df = self._converter._variables_to_df(X)
        provisions = {st: self._provision_adapter.calculate_provision(st, vars_df) for st in self._chosen_service_types}

        if np.count_nonzero(x) != 0:
            self._init_provisions = provisions

        return provisions

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
        demand_ub = self._capacity_checker.get_ub_var(var)
        return min(demand_ub, area_ub)

    def get_var_weight(self, var_num: int) -> float:
        """
        Get the weight for a variable.

        Parameters
        ----------
        var_num : int
            Variable index.

        Returns
        -------
        float
            Weight value (sum of build_floor_area and site_area).
        """
        var = self._converter.X[var_num]
        return var.build_floor_area + var.site_area

    def get_limit_var(self, var_num: int) -> float:
        """
        Get the area limit for a variable's block.

        Parameters
        ----------
        var_num : int
            Variable index.

        Returns
        -------
        float
            Area limit for the variable's block.
        """
        var = self._converter.X[var_num]
        return self._area_checker.areas[var.block_id]

    def get_var_block_id(self, var_num: int) -> int:
        """
        Get the block ID for a variable.

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

    def get_service_name(self, var_num: int) -> str:
        """
        Get the service type for a variable.

        Parameters
        ----------
        var_num : int
            Variable index.

        Returns
        -------
        str
            Service type associated with the variable.
        """
        return self._converter.X[var_num].service_type
