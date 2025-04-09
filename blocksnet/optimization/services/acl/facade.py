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
from blocksnet.optimization.services.acl.variable_adapters import BlockSolution
from blocksnet.optimization.services.common import ServicesContainer
from blocksnet.optimization.services.schemas import ServicesSchema

from .provision_adapter import ProvisionAdapter


class BlocksNetFacade:
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
        service_weights: Dict[str, float],
    ) -> None:
        """
        Initialize the BlocksNetFacade with required data structures.

        Parameters
        ----------
        accessibility_matrix : pd.DataFrame
            Matrix representing accessibility between units.
        blocks_df : gpd.GeoDataFrame
            GeoDataFrame containing block(unit) geometries and attributes.
        blocks_lu : Dict[int, LandUse]
            Dictionary mapping block(unit) IDs to land use types.
        service_weights : Dict[str, float]
            Dictionary mapping service types to their relative weights.
        """
        self._blocks_lu: Dict[int, LandUse] = blocks_lu
        self._area_checker: AreaChecker = AreaChecker(blocks_df)
        self._services_containers: Dict[str, ServicesContainer] = {}

        # Initialize service containers for each service type
        for st, weight in service_weights.items():
            services_df = blocks_df.rename(columns={f"capacity_{st}": "capacity"})[["capacity"]]
            services_df = ServicesSchema(services_df)
            self._services_containers[st] = ServicesContainer(name=st, weight=weight, services_df=services_df)

        # Determine valid service types based on land use
        blocks_service_types: Set[str] = reduce(
            lambda x, y: x | y, [set(service_types_config[lu]) for lu in blocks_lu.values()]
        )
        self._chosen_service_types: Set[str] = set(service_weights.keys()) & set(st for st in blocks_service_types)
        
        # Initialize provision adapter
        self._provision_adapter: ProvisionAdapter = ProvisionAdapter(
            blocks_lu, accessibility_matrix, blocks_df, self._services_containers
        )
        
        # Calculate initial provisions
        self._start_provisions: Dict[str, float] = {}
        for st in self._provision_adapter.provisions_dfs.keys():
            prov = self._provision_adapter.calculate_provision(st)
            if abs(prov - 1.0) < 1e-10 and st in self._chosen_service_types:
                self._chosen_service_types.remove(st)
            self._start_provisions[st] = prov

        # Initialize solution converter and checkers
        self._converter = BlockSolution(blocks_lu, self._chosen_service_types)
        self._capacity_checker: CapacityChecker = CapacityChecker(
            list(blocks_lu.keys()),
            self._provision_adapter._accessibility_matrix,
            self._provision_adapter.provisions_dfs,
        )
        self.num_params = len(self._converter)
        self._init_provisions: Dict[str, float] = {st: 0 for st in self._chosen_service_types}

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

    def check_constraints(self, x: ArrayLike) -> bool:
        """
        Check if a solution satisfies all constraints.

        Parameters
        ----------
        x : ArrayLike
            Solution array to validate.

        Returns
        -------
        bool
            True if all constraints are satisfied, False otherwise.
        """
        X = self._converter(x)
        return self._area_checker.check_constraints(X)
    
    def get_max_capacity(self, block_id: int, service: str) -> float:
        """
        Get the maximum capacity for a specific service type in a given block.
        
        This method calculates the maximum possible capacity for a service type
        in a specific block, based on the accessibility matrix and provision data.
        
        Parameters
        ----------
        block_id : int
            ID of the block to query.
        service : str
            Service type to calculate capacity for.
            
        Returns
        -------
        float
            Maximum capacity value for the specified service in the given block.
        """
        return self._capacity_checker.get_demand(block_id, service, self._provision_adapter._accessibility_matrix, self._provision_adapter.get_provision_df(service))

    def get_max_capacities(self, block_id: int) -> Dict[str, float]:
        """
        Get maximum capacities for all valid service types in a given block.
        
        This method returns a dictionary mapping each valid service type for the block
        to its maximum possible capacity value.
        
        Parameters
        ----------
        block_id : int
            ID of the block to query.
            
        Returns
        -------
        Dict[str, float]
            Dictionary mapping service types to their maximum capacity values.
        """
        services = self.get_block_services(block_id)
        return {st: self.get_max_capacity(block_id, st) for st in services}


    def update_area(self, x: ArrayLike):
        """
        Update area allocations based on the given solution.

        Parameters
        ----------
        x : ArrayLike
            Solution array containing updated area allocations.
        """
        X = self._converter(x)
        self._area_checker.update_area(X)

    def get_distance(self, x: ArrayLike) -> float:
        """
        Calculate the total distance metric for the entire solution.

        Parameters
        ----------
        x : ArrayLike
            Solution array to evaluate.

        Returns
        -------
        float
            Total distance metric for the solution.
        """
        X = self._converter(x)
        return self._area_checker.get_distance_for_entire_solution(X)

    def get_max_distance(self) -> float:
        """
        Calculate the maximum possible distance metric when all variables are at their upper bounds.

        Returns
        -------
        float
            Maximum possible distance metric.
        """
        X = copy.deepcopy(self._converter.X)
        for i, x in enumerate(X):
            X[i].count = self.get_upper_bound_var(i)
        return self._area_checker.get_distance_for_entire_solution(X)

    def get_block_services(self, block_id: int) -> Set[str]:
        """
        Get the set of valid service types for a specific block based on its land use.

        Parameters
        ----------
        block_id : int
            ID of the block to query.

        Returns
        -------
        Set[str]
            Set of service types valid for the specified block.
        """
        chosen_service_types = set(st for st in set(service_types_config[self._blocks_lu[block_id]]))
        return chosen_service_types & self._chosen_service_types

    def get_changed_services(self, x_init: ArrayLike, x: ArrayLike) -> Set[str]:
        """
        Identify which services have changed between two solutions.

        Parameters
        ----------
        x_init : ArrayLike
            Initial solution array.
        x : ArrayLike
            Current solution array.

        Returns
        -------
        Set[str]
            Set of service types that have changed between the solutions.
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
            if (
                np.count_nonzero(services_vars[st]) == 0 or np.count_nonzero(services_vars_diff[st]) == 0
            ):  # all zeros or all same as init
                if st in changed_services:
                    changed_services.remove(st)

        return set([st for st in self._chosen_service_types if st in changed_services])

    def get_provisions(self, x_init: ArrayLike, x: ArrayLike) -> Dict[str, float]:
        """
        Calculate provision values for services that have changed between two solutions.

        Parameters
        ----------
        x_init : ArrayLike
            Initial solution array.
        x : ArrayLike
            Current solution array.

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
        Calculate provision values for all services in the current solution.

        Parameters
        ----------
        x : ArrayLike
            Current solution array.

        Returns
        -------
        Dict[str, float]
            Dictionary mapping all service types to their provision values.
        """
        X = self._converter(x)
        changed_services = self._chosen_service_types

        vars_df = self._converter._variables_to_df(X)
        provisions = {st: self._provision_adapter.calculate_provision(st, vars_df) for st in changed_services}

        if np.count_nonzero(x) != 0:  # first trial (not null)
            self._init_provisions = provisions

        return provisions

    def get_upper_bound_var(self, var_num: int) -> int:
        """
        Get the upper bound for a specific variable.

        Parameters
        ----------
        var_num : int
            Index of the variable.

        Returns
        -------
        int
            Upper bound value for the variable.
        """
        var = self._converter.X[var_num]
        area_ub = self._area_checker.get_ub_var(var)
        demand_ub = self._capacity_checker.get_ub_var(var)
        return min(demand_ub, area_ub)

    def get_var_weight(self, var_num: int) -> float:
        """
        Get the weight for a specific variable.

        Parameters
        ----------
        var_num : int
            Index of the variable.

        Returns
        -------
        float
            Weight value for the variable (sum of build_floor_area and site_area).
        """
        var = self._converter.X[var_num]
        return var.build_floor_area + var.site_area

    def get_limit_var(self, var_num: int) -> float:
        """
        Get the area limit for a specific variable's block.

        Parameters
        ----------
        var_num : int
            Index of the variable.

        Returns
        -------
        float
            Area limit for the variable's block.
        """
        var = self._converter.X[var_num]
        return self._area_checker.areas[var.block_id]

    def get_var_block_id(self, var_num: int) -> int:
        """
        Get the block ID associated with a specific variable.

        Parameters
        ----------
        var_num : int
            Index of the variable.

        Returns
        -------
        int
            Block ID associated with the variable.
        """
        return self._converter.X[var_num].block_id

    def get_service_name(self, var_num: int) -> str:
        """
        Get the service type associated with a specific variable.

        Parameters
        ----------
        var_num : int
            Index of the variable.

        Returns
        -------
        str
            Service type associated with the variable.
        """
        return self._converter.X[var_num].service_type