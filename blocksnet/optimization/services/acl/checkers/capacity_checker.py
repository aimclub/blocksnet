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

    def __init__(
        self, block_ids: List[int], accessibility_matrix: pd.DataFrame, provision_dfs: Dict[str, pd.DataFrame]
    ):
        """
        Initialize the CapacityChecker with block demand data.

        Parameters
        ----------
        block_ids : List[int]
            List of block IDs to be included in the checker.
        accessibility_matrix : pd.DataFrame
            DataFrame representing accessibility between blocks, used to determine service coverage.
        provision_dfs : Dict[str, pd.DataFrame]
            Dictionary mapping service types to their provision DataFrames containing demand information.
        """
        self._demands: Dict[int, Dict[str, float]] = {block_id: {} for block_id in block_ids}
        self._left_demands: Dict[int, Dict[str, float]] = {block_id: {} for block_id in block_ids}
        
        # Calculate initial demand for each block and service type
        for block_id in block_ids:
            for st, prov_df in provision_dfs.items():
                # Calculate total demand with 20% buffer
                block_demand = self.get_demand(block_id, st, accessibility_matrix, prov_df) * 1.2
                
                self._demands[block_id][st] = block_demand
                self._left_demands[block_id][st] = block_demand

    def get_demand(self, block_id: int, service: str, accessibility_matrix: pd.DataFrame, provision_df: pd.DataFrame) -> float:
                # Get accessibility threshold for this service type
        _, _, accessibility = service_types_config[service].values()
                
        # Find blocks within accessibility range
        nearest_blocks = accessibility_matrix.index[accessibility_matrix[block_id] <= accessibility]
        
        # Calculate total demand with 20% buffer
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
        block_demands = self._demands.copy()
        
        for var in X:
            # Subtract this variable's capacity from the remaining demand
            block_demands[var.block_id][var.service_type] -= var.total_capacity
            
            # Check if we've exceeded capacity for this service type
            if block_demands[var.block_id][var.service_type] < 0:
                return False
                
        return True