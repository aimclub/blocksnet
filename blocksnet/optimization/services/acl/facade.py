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
from blocksnet.optimization.services.common.variable import Variable
from blocksnet.optimization.services.schemas import ServicesSchema
from blocksnet.relations.accessibility import validate_accessibility_matrix

from .provision_adapter import ProvisionAdapter


class Facade:
    """
    A facade class providing a unified interface for urban service allocation optimization.

    This class coordinates between various components including:
    - Variable adapters for solution representation
    - Capacity and area constraint checkers
    - Provision calculation and validation
    - Service type management

    It serves as the main interface between optimization algorithms and the urban service model.
    """

    def __init__(
        self,
        var_adapter: VariableAdapter,
        accessibility_matrix: pd.DataFrame,
        blocks_df: gpd.GeoDataFrame,
        blocks_lu: Dict[int, LandUse],
    ) -> None:
        """
        Initialize the Facade with urban data and optimization components.

        Parameters
        ----------
        var_adapter : VariableAdapter
            Adapter for converting between solution vectors and service variables.
        accessibility_matrix : pd.DataFrame
            Square matrix where entry (i,j) represents accessibility from block i to service j.
        blocks_df : gpd.GeoDataFrame
            GeoDataFrame containing block geometries and attributes. Must include:
            - 'geometry' column with polygon geometries
            - Columns matching accessibility_matrix indices
        blocks_lu : Dict[int, LandUse]
            Dictionary mapping block IDs to land use types (LandUse enum values).

        Raises
        ------
        ValueError
            If accessibility_matrix and blocks_df dimensions don't match.
        """
        validate_accessibility_matrix(accessibility_matrix, blocks_df)
        self._blocks_lu = blocks_lu
        self._area_checker = AreaChecker(blocks_lu, blocks_df)

        # Initialize service type configuration
        blocks_service_types = reduce(lambda x, y: x | y, [set(service_types_config[lu]) for lu in blocks_lu.values()])
        self._blocks_service_types = set(blocks_service_types)
        self._chosen_service_types = set()

        # Initialize provision calculation components
        self._provision_adapter = ProvisionAdapter(blocks_lu, accessibility_matrix, blocks_df)
        self._start_provisions = {}
        self._last_provisions = {}

        # Initialize optimization components
        self._converter = var_adapter
        self._capacity_checker = CapacityChecker(
            list(blocks_lu.keys()),
            self._provision_adapter._accessibility_matrix,
        )
        self.num_params = 0

    @property
    def start_provisions(self) -> Dict[str, float]:
        """
        Get the initial provision levels before optimization.

        Returns
        -------
        Dict[str, float]
            Dictionary mapping service type names to their initial provision values.
            Provision values range from 0 to 1, where 1 represents full coverage.
        """
        return self._start_provisions

    @property
    def last_provisions(self) -> Dict[str, float]:
        """
        Get the provision levels from the last evaluated solution.

        Returns
        -------
        Dict[str, float]
            Dictionary mapping service type names to their most recent provision values.
            Useful for tracking changes during optimization.
        """
        return self._last_provisions

    def solution_to_services_df(self, solution: Dict[str, int]) -> pd.DataFrame:
        """
        Convert an optimization solution to a structured service allocation DataFrame.

        Parameters
        ----------
        solution : Dict[str, int]
            Dictionary mapping variable names (e.g., 'x0', 'x1') to their values.

        Returns
        -------
        pd.DataFrame
            DataFrame with service allocation details containing columns:
            - block_id: Integer block identifier
            - service_type: String service type name
            - site_area: Numeric area occupied (square meters)
            - build_floor_area: Numeric floor area (square meters)
            - capacity: Numeric service capacity (units)
            - count: Integer number of units allocated
        """
        x = np.zeros(self.num_params)
        for var_name, var_val in solution.items():
            x[int(var_name[2:])] = var_val
        X = self._converter(x)

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
        df = pd.DataFrame(xs)
        return df[df["count"] != 0] if len(xs) > 0 else df

    def add_service_type(self, name: str, weight: float, services_df: pd.DataFrame) -> None:
        """
        Register a service type for optimization consideration.

        Parameters
        ----------
        name : str
            Unique identifier for the service type (must match config).
        weight : float
            Relative importance weight for this service in optimization.
        services_df : pd.DataFrame
            DataFrame containing service specifications with columns:
            - service_type: String matching 'name' parameter
            - site_area: Numeric area requirements
            - build_floor_area: Numeric floor area requirements
            - capacity: Numeric service capacity per unit

        Notes
        -----
        Only adds the service if it's compatible with existing land uses
        and has non-trivial provision values (≠1).
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
        if abs(prov - 1.0) > 1e-10:
            self._capacity_checker.add_service_type(name, provision_df)
            self._converter.add_service_type_vars(name)
            self.num_params = len(self._converter)
            self._chosen_service_types.add(name)

        self._start_provisions[name] = prov
        self._last_provisions[name] = prov

    def check_constraints(self, x: ArrayLike) -> bool:
        """
        Validate a solution against all urban planning constraints.

        Parameters
        ----------
        x : ArrayLike
            1D numpy array representing the solution vector.

        Returns
        -------
        bool
            True if solution satisfies both area and capacity constraints,
            False otherwise.
        """
        X = self._converter(x)
        return self._area_checker.check_constraints(X) and self._capacity_checker.check_constraints(X)

    def get_X(self, x: ArrayLike) -> List[Variable]:
        """
        Convert solution vector to Variable objects.

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
        return self._converter.X

    def get_max_capacity(self, block_id: int, service: str) -> float:
        """
        Get maximum service capacity for a specific block and service type.

        Parameters
        ----------
        block_id : int
            Identifier of the urban block.
        service : str
            Service type identifier.

        Returns
        -------
        float
            Maximum allowable capacity considering demand constraints.
        """
        return self._capacity_checker._demands[block_id][service]

    def get_max_capacities(self, block_id: int) -> Dict[str, float]:
        """
        Get maximum capacities for all services in a block.

        Parameters
        ----------
        block_id : int
            Identifier of the urban block.

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
            Summed distance metric representing solution quality.
        """
        X = self._converter(x)
        return self._area_checker.get_distance_for_entire_solution(X)

    def get_max_distance(self) -> float:
        """
        Calculate theoretical maximum distance metric.

        Returns
        -------
        float
            Maximum possible distance value when all variables are at upper bounds.
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
            Identifier of the urban block.

        Returns
        -------
        Set[str]
            Set of service type identifiers compatible with the block's land use.
        """
        land_use_services = set(service_types_config[self._blocks_lu[block_id]])
        return land_use_services & self._chosen_service_types

    def get_changed_services(self, x_last: ArrayLike, x: ArrayLike) -> Set[str]:
        """
        Identify services with changed allocations between solutions.

        Parameters
        ----------
        x_last : ArrayLike
            Previous solution vector.
        x : ArrayLike
            Current solution vector.

        Returns
        -------
        Set[str]
            Set of service type identifiers that changed allocation.
        """
        X_diff = self._converter(x - x_last)
        services_vars_diff = {
            st: np.array([var.count for var in X_diff if var.service_type == st and var.count > 0])
            for st in self._chosen_service_types
        }

        X = self._converter(x)
        services_vars = {
            st: np.array([var.count for var in X if var.service_type == st and var.count > 0])
            for st in self._chosen_service_types
        }

        changed_services = {
            st for st in self._chosen_service_types if len(services_vars[st]) > 0 and len(services_vars_diff[st]) > 0
        }
        return changed_services

    def get_provisions(self, x_last: ArrayLike, x: ArrayLike) -> Dict[str, float]:
        """
        Calculate provision levels for changed services between solutions.

        Parameters
        ----------
        x_last : ArrayLike
            Previous solution vector.
        x : ArrayLike
            Current solution vector.

        Returns
        -------
        Dict[str, float]
            Dictionary mapping service types to updated provision levels.
        """
        X = self._converter(x)
        changed_services = self.get_changed_services(x_last, x)

        vars_df = self._converter._variables_to_df(X)
        provisions = {st: self._provision_adapter.calculate_provision(st, vars_df) for st in changed_services}

        # Maintain provisions for unchanged services
        for st in self._last_provisions.keys() - changed_services:
            provisions[st] = (
                self._start_provisions[st]
                if sum(var.count for var in X if var.service_type == st) == 0
                else self._last_provisions[st]
            )

        self._last_provisions = provisions
        return provisions

    def get_all_provisions(self, x: ArrayLike) -> Dict[str, float]:
        """
        Calculate provision levels for all services in a solution.

        Parameters
        ----------
        x : ArrayLike
            Solution vector to evaluate.

        Returns
        -------
        Dict[str, float]
            Complete dictionary of service provisions.
        """
        X = self._converter(x)
        vars_df = self._converter._variables_to_df(X)
        provisions = {st: self._provision_adapter.calculate_provision(st, vars_df) for st in self._chosen_service_types}
        if np.count_nonzero(x) != 0:
            self._last_provisions = provisions
        return provisions

    def get_upper_bound_var(self, var_num: int) -> int:
        """
        Get upper bound for a specific optimization variable.

        Parameters
        ----------
        var_num : int
            Index of the variable in the solution vector.

        Returns
        -------
        int
            Maximum allowed value considering both area and capacity constraints.
        """
        var = self._converter.X[var_num]
        area_ub = self._area_checker.get_ub_var(var)
        demand_ub = self._capacity_checker.get_ub_var(var)
        return min(filter(lambda x: x != -1, [area_ub, demand_ub]), default=-1)

    def get_var_weights(self, var_num: int) -> ArrayLike:
        """
        Get weight components for a variable.

        Parameters
        ----------
        var_num : int
            Index of the variable in the solution vector.

        Returns
        -------
        ArrayLike
            Array containing [site_area, build_floor_area, capacity] weights.
        """
        var = self._converter.X[var_num]
        return np.array([var.site_area, var.build_floor_area, var.capacity])

    def get_limits_var(self, var_num: int) -> ArrayLike:
        """
        Get resource limits for a variable's block.

        Parameters
        ----------
        var_num : int
            Index of the variable in the solution vector.

        Returns
        -------
        ArrayLike
            Array containing [site_area_limit, build_floor_area_limit, capacity_limit].
        """
        var = self._converter.X[var_num]
        return np.array(
            [
                self._area_checker.site_areas[var.block_id],
                self._area_checker.build_floor_areas[var.block_id],
                self._capacity_checker._demands[var.block_id][var.service_type],
            ]
        )

    def get_var_block_id(self, var_num: int) -> int:
        """
        Get block ID associated with a variable.

        Parameters
        ----------
        var_num : int
            Index of the variable in the solution vector.

        Returns
        -------
        int
            Block identifier.
        """
        return self._converter.X[var_num].block_id

    def get_service_name(self, var_num: int) -> str:
        """
        Get service type name for a variable.

        Parameters
        ----------
        var_num : int
            Index of the variable in the solution vector.

        Returns
        -------
        str
            Service type identifier.
        """
        return self._converter.X[var_num].service_type
