from typing import Dict, List

import numpy as np

from blocksnet.method.annealing_optimizer import Variable


class AreaChecker:
    """
    Class that computes distances related to the acceptance of variables in different blocks.

    This class evaluates the distance for a given set of variables relative to specific blocks. It provides
    methods for calculating the distance to acceptance variables for a given block and for the entire solution.
    """

    def get_distance_by_block(self, X: List[Variable], block_id: int) -> float:
        """
        Calculates the distance for a specific block relative to the acceptance variables.

        This method computes a weighted distance for a specific block (identified by `block_id`) using
        the area and values of the variables assigned to that block.

        Parameters
        ----------
        X : List[Variable]
            List of variables representing the current solution.
        block_id : int
            The identifier of the block for which the distance should be calculated.

        Returns
        -------
        float
            The calculated distance for the specified block.
        """
        block_denom = 0
        block_numer = 0
        dist = 0

        # Iterate over all variables in the solution
        # Calculate the numerator and denominator for the distance metric
        for x in X:
            if x.block.id != block_id:
                continue  # Skip variables not belonging to the current block
            block_numer += x.brick.area * x.value  # Numerator: area of the brick times its value
            block_denom += x.brick.area * x.brick.area  # Denominator: area of the brick squared

        # If the denominator is positive, calculate the distance
        if block_numer > 0:
            dist += block_numer / np.sqrt(block_denom)

        return dist

    def get_distance_for_entire_solution(self, X: List[Variable]) -> float:
        """
        Calculates the total distance for the entire solution.

        This method calculates the sum of distances for all blocks in the solution, calling the method
        `get_distance_for_entire_solution_block` for each unique block.

        Parameters
        ----------
        X : List[Variable]
            List of variables representing the current solution.

        Returns
        -------
        float
            The total distance of the solution, calculated as the sum of distances for all blocks.
        """
        block_ids = set(x.block.id for x in X)  # Get unique block IDs in the solution
        total_distance = 0

        # Sum up the distances for all blocks in the solution
        for block_id in block_ids:
            total_distance += self.get_distance_by_block(X, block_id)

        return total_distance
