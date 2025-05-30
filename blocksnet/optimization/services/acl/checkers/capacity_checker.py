import copy
from typing import Dict, List

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
            # Calculate total demand with 20% buffer
            block_demand = self.get_demand(block_id, st, provision_df)
            units = service_types_config.units
            block_demand += max([unit.capacity for _, unit in units[units.service_type == st].iterrows()])

            self._demands[block_id][st] = max(block_demand, 0)

    def get_demand(self, block_id: int, service: str, provision_df: pd.DataFrame) -> float:
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

    def get_ub_var(self, var: Variable) -> int:
        """
        Calculate the upper bound (maximum count) for a service variable based on remaining demand capacity.

        Parameters
        ----------
        var : Variable
            The service variable to calculate upper bound for.

        Returns
        -------
        int
            Maximum possible count for the variable given remaining demand capacity.
        """
        block_demand = self._demands[var.block_id][var.service_type]
        return int(np.floor(block_demand / var.capacity))

    def check_constraints(self, X: List[Variable]) -> bool:
        """
        Verify that the solution doesn't exceed demand capacity constraints for any service type.

        Parameters
        ----------
        X : List[Variable]
            List of Variable objects representing the solution.

        Returns
        -------
        bool
            True if all service capacities in the solution satisfy demand constraints,
            False if any service exceeds its demand capacity.
        """
        # Create a copy of demands to track remaining capacity
        block_demands = copy.deepcopy(self._demands)

        for var in X:
            # Subtract this variable's capacity from the remaining demand
            block_demands[var.block_id][var.service_type] -= var.total_capacity

            # Check if we've exceeded capacity for this service type
            if block_demands[var.block_id][var.service_type] < 0:
                return False

        return True
