from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from numpy.typing import ArrayLike

from blocksnet.config.service_types.config import service_types_config
from blocksnet.enums import LandUse
from blocksnet.optimization.services.common.variable import Variable

from .variable_adapter import VariableAdapter


class AreaVariable(Variable):
    """A specialized Variable class for handling area-based service allocation."""

    pass


@dataclass
class Unit:
    """
    Data class representing a service unit with area and capacity information.

    Attributes
    ----------
    site_area : int
        The area occupied by the unit on the site.
    build_floor_area : int
        The floor area of the built unit.
    capacity : int
        The service capacity provided by the unit.
    """

    site_area: int
    build_floor_area: int
    capacity: int


class AreaSolution(VariableAdapter):
    """
    A class for managing area-based service allocation solutions.

    Handles conversion between area-based variables and unit-based variables,
    and manages the mapping between solution representations.
    """

    def __init__(self, blocks_lu: Dict[int, LandUse]):
        """
        Initialize the AreaSolution with land use data for blocks.

        Parameters
        ----------
        blocks_lu : Dict[int, LandUse]
            Dictionary mapping block IDs to their land use types.
        """
        super().__init__()
        self._blocks_lu: Dict[int, LandUse] = blocks_lu
        self._areas_X: List[AreaVariable] = []
        self._units_lists: Dict[str, List[Unit]] = {}
        self._build_floor_area_coef = 1.0

    def add_service_type_vars(self, service_type: str):
        """
        Add variables for a specific service type across all blocks.

        Parameters
        ----------
        service_type : str
            The service type identifier to add variables for.
        """
        for block_id, land_use in self._blocks_lu.items():
            self.add_variables(block_id, land_use, service_type)

    @property
    def X(self) -> List[Variable]:
        """
        Get the list of variables representing the current solution.

        Returns
        -------
        List[Variable]
            The list of variables representing the current solution.
        """
        return self._areas_X

    def add_variables(self, block_id: int, land_use: LandUse, service_type: str):
        """
        Add variables for a specific block, land use, and service type.

        Parameters
        ----------
        block_id : int
            The ID of the block to add variables for.
        land_use : LandUse
            The land use type of the block.
        service_type : str
            The service type identifier to add variables for.
        """
        if service_type not in service_types_config[land_use]:
            return

        # Create base area variable
        x_area = AreaVariable(block_id=block_id, service_type=service_type, site_area=1, build_floor_area=0, capacity=0)

        # Get all units for this service type
        units = service_types_config.units
        units = units[units.service_type == service_type]
        self._units_lists.update({service_type: []})

        # Create variables and units for each configuration
        for _, unit in units.iterrows():
            x = Variable(block_id=block_id, **unit)
            u = Unit(site_area=x.site_area, build_floor_area=x.build_floor_area, capacity=x.capacity)
            self._units_lists[service_type].append(u)
            self._X.append(x)

        # Sort units by capacity per combined area
        def capacity_area_sortkey(u: Unit):
            return -u.capacity / (u.build_floor_area * self._build_floor_area_coef + u.site_area)

        self._units_lists[service_type].sort(key=capacity_area_sortkey)

        # Set the area variable's capacity based on the best unit
        best_unit = self._units_lists[service_type][0]
        x_area.capacity = best_unit.capacity / (best_unit.build_floor_area + best_unit.site_area)
        self._areas_X.append(x_area)

    def _inject_area_to_X(self, x_area: AreaVariable):
        """
        Distribute an area variable's allocation across concrete unit variables.

        Parameters
        ----------
        x_area : AreaVariable
            The area variable to distribute to concrete units.
        """
        left_area = x_area.total_site_area

        # Reset counts for all variables of this block and service type
        for x in self._X:
            if x.block_id == x_area.block_id and x.service_type == x_area.service_type:
                x.count = 0

        # Distribute the area to units in priority order
        for u in self._units_lists[x_area.service_type]:
            for x in self._X:
                if (
                    x.block_id == x_area.block_id
                    and x.service_type == x_area.service_type
                    and abs(x.site_area - u.site_area) < 1e-6
                    and (x.build_floor_area - u.build_floor_area) < 1e-6
                    and x.capacity == u.capacity
                ):
                    x.count = int(np.floor(left_area / (x.build_floor_area + x.site_area)))
                    left_area -= x.total_build_floor_area + x.total_site_area
                    break
        x_area.count -= left_area

    def _inject_solution_to_X(self, solution: ArrayLike):
        """
        Map a numpy array solution vector to the internal variables.

        Parameters
        ----------
        solution : ArrayLike
            Numpy array representing the solution vector.

        Raises
        ------
        ValueError
            If the solution is not a 1D array or contains NaN values.
        ArrayLengthError
            If the solution length doesn't match the number of variables.
        """
        if len(solution.shape) != 1:
            raise ValueError("Solution must be a 1D array.")

        # Update each area variable and distribute to concrete units
        for var, val in zip(self._areas_X, solution):
            var.count = int(val)
            self._inject_area_to_X(var)
