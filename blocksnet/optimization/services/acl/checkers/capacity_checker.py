import copy
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from blocksnet.analysis.provision.competivive.core import DEMAND_LEFT_COLUMN
from blocksnet.config.service_types.config import service_types_config
from blocksnet.optimization.services.common.variable import Variable


class CapacityChecker:
    """
    A class for checking and managing service capacity constraints in urban block optimization problems.

    This class tracks service demand capacities, calculates upper bounds for service variables,
    and verifies that solutions don't exceed demand capacities for different service types.
    """

    def __init__(self, block_ids: List[int], accessibility_matrix: pd.DataFrame):
        """
        Initialize the CapacityChecker with block demand data.

        Parameters
        ----------
        block_ids : List[int]
            List of block IDs to be included in the checker.
        accessibility_matrix : pd.DataFrame
            DataFrame representing accessibility between blocks, used to determine service coverage.
        """
        self._demands: Dict[int, Dict[str, float]] = {block_id: {} for block_id in block_ids}
        self._accessibility_matrix = accessibility_matrix

    def add_service_type(self, st: str, provision_df: pd.DataFrame):
        """
        Add a service type and its provision data to the capacity checker.

        Parameters
        ----------
        st : str
            Service type identifier to be added.
        provision_df : pd.DataFrame
            DataFrame containing provision data for the service type, including demand information.
        """
        for block_id in self._demands.keys():
            block_demand = self.calc_demand(block_id, st, provision_df)
            units = service_types_config.units

            if any(units[units.service_type == st].capacity >= block_demand):
                block_demand += min(
                    [
                        unit.capacity
                        for _, unit in units[units.service_type == st].iterrows()
                        if unit.capacity >= block_demand
                    ]
                )
            else:
                block_demand += max(
                    [
                        unit.capacity
                        for _, unit in units[units.service_type == st].iterrows()
                        if unit.capacity <= block_demand
                    ]
                )

            self._demands[block_id][st] = max(block_demand, 0)

    def calc_demand(self, block_id: int, service: str, provision_df: pd.DataFrame) -> float:
        """
        Calculate the total demand for a service in a given block's accessible area.

        Parameters
        ----------
        block_id : int
            ID of the block to calculate demand for.
        service : str
            Service type identifier.
        provision_df : pd.DataFrame
            DataFrame containing provision data for the service.

        Returns
        -------
        float
            Total calculated demand for the service in the accessible area around the block.
        """
        # Get accessibility threshold for this service type
        _, _, accessibility = service_types_config[service].values()

        # Find blocks within accessibility range
        nearest_blocks = self._accessibility_matrix.index[self._accessibility_matrix[block_id] <= accessibility]

        return provision_df[provision_df.index.isin(nearest_blocks)][DEMAND_LEFT_COLUMN].sum()

    def get_demand(self, block_id: int, service: str, demand: int = 0) -> float:
        """
        Retrieve the demand for a specific service in a block, optionally adjusted by additional population.

        Parameters
        ----------
        block_id : int
            ID of the block to get demand for.
        service : str
            Service type identifier.
        demand : int, optional
            Additional demand to add to the base demand (default is 0).

        Returns
        -------
        float
            Total demand for the service in the block, including any additional population.
        """
        return self._demands[block_id][service] + demand

    def get_ub_var(self, var: Variable, demand: int = 0) -> int:
        """
        Calculate the upper bound (maximum count) for a service variable based on remaining demand capacity.

        Parameters
        ----------
        var : Variable
            The service variable to calculate upper bound for.
        demand : int, optional
            Additional demand to consider in demand calculation (default is 0).

        Returns
        -------
        int
            Maximum possible count for the variable given remaining demand capacity.
            Returns -1 if the variable has zero capacity.
        """
        block_demand = self.get_demand(var.block_id, var.service_type, demand)
        if var.capacity == 0:
            return -1
        return int(np.floor(block_demand / var.capacity))

    def check_constraints(
        self, X: List[Variable], blocks_services_demand: Optional[Dict[int, Dict[str, int]]] = None
    ) -> bool:
        """
        Verify that the solution doesn't exceed demand capacity constraints for any service type.

        Parameters
        ----------
        X : List[Variable]
            List of Variable objects representing the solution.
        blocks_services_demand : Optional[Dict[int, Dict[str, int]]], optional
            Additional demand per service type per block to consider in constraints (default is None).

        Returns
        -------
        bool
            True if all service capacities in the solution satisfy demand constraints,
            False if any service exceeds its demand capacity.
        """
        # Create a copy of demands to track remaining capacity
        block_demands = copy.deepcopy(self._demands)
        if blocks_services_demand is not None:
            # Update block demands with additional demand from the solution
            for block_id, services in blocks_services_demand.items():
                for service, demand in services.items():
                    if service in block_demands[block_id]:
                        block_demands[block_id][service] += demand

        for var in X:
            # Subtract this variable's capacity from the remaining demand
            block_demands[var.block_id][var.service_type] -= var.total_capacity

            # Check if we've exceeded capacity for this service type
            if block_demands[var.block_id][var.service_type] < 0:
                return False

        return True
