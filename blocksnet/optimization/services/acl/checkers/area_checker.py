from typing import Dict, List, Set

import numpy as np
import pandas as pd

from blocksnet.config.land_use.common.fsi_ranges import fsi_ranges
from blocksnet.config.land_use.common.gsi_ranges import gsi_ranges
from blocksnet.enums import LandUse
from blocksnet.optimization.services.common.variable import Variable


class AreaChecker:
    """
    A class for checking and managing area constraints in urban block optimization problems.

    This class tracks available areas in blocks, calculates distance metrics for solutions,
    and verifies that solutions don't exceed available areas. It handles two types of areas:
    - Site area (ground space)
    - Build floor area (vertical space)

    The class uses FSI (Floor Space Index) and GSI (Ground Space Index) ranges from configuration
    to determine maximum allowable development for different land use types.
    """

    def __init__(self, blocks_lu: Dict[int, LandUse], blocks_df: pd.DataFrame) -> None:
        """
        Initialize the AreaChecker with block area data and land use information.

        Parameters
        ----------
        blocks_lu : Dict[int, LandUse]
            Dictionary mapping block IDs to their land use types.
        blocks_df : pd.DataFrame
            DataFrame containing block information, must include:
            - 'site_area' column: Ground area available for each block
            - Index should be block IDs matching keys in blocks_lu

        Notes
        -----
        Initializes two area dictionaries:
        - site_areas: Available ground space per block (adjusted by GSI)
        - build_floor_areas: Available vertical space per block (adjusted by FSI)
        """
        self._bfa_coef = 0.3  # Coefficient for residential build floor area adjustment
        self._sa_coef = 0.8  # Coefficient for site area adjustment
        self.site_areas = {block_id: cols["site_area"] for block_id, cols in blocks_df.iterrows()}
        self.build_floor_areas = {block_id: cols["site_area"] for block_id, cols in blocks_df.iterrows()}

        # Adjust areas based on land use regulations
        for block_id, land_use in blocks_lu.items():
            alpha = fsi_ranges[land_use][1] * (
                self._bfa_coef if land_use == LandUse.RESIDENTIAL else 1.0
            )  # fsi_max * bfa_coef
            beta = alpha - gsi_ranges[land_use][0] + self._sa_coef  # alpha - gsi_min + sa_coef
            self.site_areas[block_id] *= beta
            self.build_floor_areas[block_id] *= alpha

    def get_distance_by_block(self, X: List[Variable], block_id: int) -> float:
        """
        Calculate the utilization efficiency metric for a specific block in the solution.

        The distance metric represents how well the solution utilizes the available area
        in the specified block. Higher values (closer to 1) indicate better utilization.

        Parameters
        ----------
        X : List[Variable]
            List of Variable objects representing the current solution.
        block_id : int
            ID of the block to calculate distance for.

        Returns
        -------
        float
            Utilization efficiency metric (0 if no variables in block, otherwise between 0-1).
            Calculated as: sum(areas) / sqrt(sum(areas^2))
        """
        block_denom = 0  # Sum of squared areas (denominator)
        block_numer = 0  # Sum of areas (numerator)
        dist = 0

        for var in X:
            if var.block_id != block_id:
                continue  # Skip variables not in current block

            block_numer += var.total_build_floor_area + var.total_site_area
            block_denom += (var.build_floor_area + var.site_area) ** 2

        if block_numer > 0:
            dist += block_numer / np.sqrt(block_denom)

        return dist

    def get_ub_var(self, var: Variable) -> int:
        """
        Calculate the maximum allowable count for a variable based on remaining area.

        Parameters
        ----------
        var : Variable
            The service variable to calculate upper bound for. Must have:
            - block_id: ID of the block
            - site_area: Ground area required per unit
            - build_floor_area: Vertical area required per unit

        Returns
        -------
        int
            Maximum possible count for the variable given remaining area constraints.
            Returns floor of minimum between site and build floor area constraints.
        """
        if var.site_area == 0:  # Only build floor area constraint applies
            return int(np.floor(self.build_floor_areas[var.block_id] / var.build_floor_area))
        if var.build_floor_area == 0:  # Only site area constraint applies
            return int(np.floor(self.site_areas[var.block_id] / var.site_area))

        # Both constraints apply - take minimum
        return min(
            int(np.floor(self.site_areas[var.block_id] / var.site_area)),
            int(np.floor(self.build_floor_areas[var.block_id] / var.build_floor_area)),
        )

    def get_distance_for_entire_solution(self, X: List[Variable]) -> float:
        """
        Calculate the total utilization efficiency metric for the complete solution.

        Parameters
        ----------
        X : List[Variable]
            List of Variable objects representing the complete solution across all blocks.

        Returns
        -------
        float
            Sum of utilization metrics for all blocks in the solution.
            Higher values indicate better overall area utilization.
        """
        block_ids: Set[int] = set(var.block_id for var in X)
        total_distance = 0.0

        for block_id in block_ids:
            total_distance += self.get_distance_by_block(X, block_id)

        return total_distance

    def check_constraints(self, X: List[Variable]) -> bool:
        """
        Verify that the solution satisfies all area constraints across all blocks.

        Parameters
        ----------
        X : List[Variable]
            List of Variable objects representing the complete solution.

        Returns
        -------
        bool
            True if all blocks satisfy both site area and build floor area constraints,
            False if any block exceeds either area limit.
        """
        block_ids: Set[int] = set(var.block_id for var in X)
        block_sums = {block_id: [0.0, 0.0] for block_id in block_ids}  # [site_area, build_floor_area]

        # Aggregate total used areas per block
        for var in X:
            block_sums[var.block_id][0] += var.total_site_area
            block_sums[var.block_id][1] += var.total_build_floor_area

        # Check all blocks against their area limits
        return all(
            (block_sums[block_id][0] <= self.site_areas[block_id])
            and (block_sums[block_id][1] <= self.build_floor_areas[block_id])
            for block_id in block_ids
        )
