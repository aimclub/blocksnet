from typing import List

import numpy as np
import pandas as pd

from blocksnet.optimization.services.common.variable import Variable


class AreaChecker:
    """
    A class for checking and managing area constraints in urban block optimization problems.

    This class tracks available areas in blocks, calculates distance metrics for solutions,
    and verifies that solutions don't exceed available areas.
    """

    def __init__(self, blocks_df: pd.DataFrame) -> None:
        """
        Initialize the AreaChecker with block area data.

        Parameters
        ----------
        blocks_df : pd.DataFrame
            DataFrame containing block information, must include 'site_area' column.
        """
        self.areas = {block_id: cols["site_area"] for block_id, cols in blocks_df.iterrows()}
        self._left_areas = {block_id: cols["site_area"] for block_id, cols in blocks_df.iterrows()}

    def get_distance_by_block(self, X: List[Variable], block_id: int) -> float:
        """
        Calculate the distance metric for a specific block in the solution.

        The distance metric represents how well the solution utilizes the available area
        in the specified block. Higher values indicate better utilization.

        Parameters
        ----------
        X : List[Variable]
            List of Variable objects representing the current solution.
        block_id : int
            ID of the block to calculate distance for.

        Returns
        -------
        float
            Distance metric for the specified block (0 if no variables in block).
        """
        block_denom = 0
        block_numer = 0
        dist = 0

        # Iterate over all variables in the solution
        for var in X:
            if var.block_id != block_id:
                continue  # Skip variables not belonging to the current block

            # Calculate numerator: sum of (area * count) for all variables in block
            block_numer += var.total_build_floor_area + var.total_site_area

            # Calculate denominator: sum of (area^2) for all variables in block
            block_denom += (var.build_floor_area + var.site_area) ** 2

        # Only calculate distance if there are variables in this block
        if block_numer > 0:
            dist += block_numer / np.sqrt(block_denom)

        return dist

    def get_ub_var(self, var: Variable) -> int:
        """
        Get the upper bound (maximum count) for a variable based on remaining area.

        Parameters
        ----------
        var : Variable
            The variable to calculate upper bound for.

        Returns
        -------
        int
            Maximum possible count for the variable given remaining area.
        """
        return int(np.floor(self._left_areas[var.block_id] / (var.site_area + var.build_floor_area)))

    def update_area(self, X: List[Variable]) -> None:
        """
        Update remaining areas after applying a solution.

        Parameters
        ----------
        X : List[Variable]
            List of Variable objects representing the solution to apply.
        """
        for var in X:
            self._left_areas[var.block_id] -= var.total_site_area + var.total_build_floor_area

    def get_distance_for_entire_solution(self, X: List[Variable]) -> float:
        """
        Calculate the total distance metric for the entire solution across all blocks.

        Parameters
        ----------
        X : List[Variable]
            List of Variable objects representing the solution.

        Returns
        -------
        float
            Sum of distance metrics for all blocks in the solution.
        """
        block_ids = set(var.block_id for var in X)  # Get unique block IDs in the solution
        total_distance = 0

        # Sum up the distances for all blocks in the solution
        for block_id in block_ids:
            total_distance += self.get_distance_by_block(X, block_id)

        return total_distance

    def check_constraints(self, X: List[Variable]) -> bool:
        """
        Verify that the solution doesn't exceed area constraints in any block.

        Parameters
        ----------
        X : List[Variable]
            List of Variable objects representing the solution.

        Returns
        -------
        bool
            True if all blocks in the solution satisfy area constraints, False otherwise.
        """
        block_ids = set(var.block_id for var in X)
        block_sums = {block_id: 0 for block_id in block_ids}

        # Calculate total used area for each block
        for var in X:
            block_sums[var.block_id] += var.total_build_floor_area + var.total_site_area

        # Check if used area <= available area for all blocks
        return all(block_sums[block_id] <= self.areas[block_id] for block_id in block_ids)
