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
        pass

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
        pass
