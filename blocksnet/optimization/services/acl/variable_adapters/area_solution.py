import copy
import logging
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from numpy.typing import ArrayLike

from blocksnet.config.service_types.config import service_types_config
from blocksnet.enums import LandUse
from blocksnet.optimization.services.common.variable import Variable

from .variable_adapter import VariableAdapter


class AreaVariable(Variable):
    """A specialized Variable class for representing area-related optimization variables."""

    pass


@dataclass
class Unit:
    """
    A dataclass representing a service unit with area and capacity attributes.

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
    A class for managing and adapting area-based service solutions in urban optimization problems.

    This class handles the conversion between different representations of service solutions,
    including area-based variables and capacity constraints.
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
        self._prev_blocks_bfa: Dict[int, float] = {}

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
        Get the list of variables representing the current city-level data.

        Returns
        -------
        List[Variable]
            The list of variables representing the current solution.
        """
        return self._areas_X

    def add_variables(self, block_id: int, land_use: LandUse, service_type: str):
        """
        Add optimization variables for a specific block, land use, and service type.

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

        x_area = AreaVariable(block_id=block_id, service_type=service_type, site_area=1, build_floor_area=0, capacity=0)

        units = service_types_config.units

        units = units[units.service_type == service_type]
        self._units_lists.update({service_type: []})

        for _, unit in units.iterrows():
            x = Variable(block_id=block_id, **unit)
            u = Unit(site_area=x.site_area, build_floor_area=x.build_floor_area, capacity=x.capacity)
            self._units_lists[service_type].append(u)
            self._X.append(x)

        def capacity_area_sortkey(u: Unit):
            if u.site_area == 0:
                return 1 / u.build_floor_area
            return -u.capacity / u.site_area

        self._units_lists[service_type].sort(key=capacity_area_sortkey)

        if np.count_nonzero(np.array([u.site_area for u in self._units_lists[service_type]])) > 0:
            x_area.capacity = min(
                [u.capacity / u.site_area for u in self._units_lists[service_type] if u.site_area > 0]
            )
        self._areas_X.append(x_area)

    def _inject_area_to_X(self, x_area: AreaVariable):
        """
        Inject area-based solution components into the main variable list.

        Parameters
        ----------
        x_area : AreaVariable
            The area variable to inject into the solution representation.
        """
        for u in self._units_lists[x_area.service_type]:
            for x in self._X:
                if (
                    x.block_id == x_area.block_id
                    and x.service_type == x_area.service_type
                    and abs(x.site_area - u.site_area) < 1e-6
                    and (x.build_floor_area - u.build_floor_area) < 1e-6
                    and x.capacity == u.capacity
                ):
                    addition = 1e9
                    if x.capacity != 0:
                        addition = int(np.floor(self._blocks_cap_cp[x.block_id][x.service_type] / x.capacity))
                    if x.build_floor_area != 0:
                        addition = min(addition, int(np.floor(self._blocks_bfa_cp[x.block_id] / x.build_floor_area)))
                    if x.site_area != 0:
                        addition = min(
                            addition, int(np.floor(self._blocks_sa[x.block_id][x.service_type] / x.site_area))
                        )
                    x.count += addition
                    self._blocks_sa[x.block_id][x.service_type] -= x.site_area * addition
                    self._blocks_bfa_cp[x.block_id] -= x.build_floor_area * addition
                    self._blocks_cap_cp[x.block_id][x.service_type] -= x.capacity * addition
                    break

    def _inject_solution_to_X(self, solution: ArrayLike):
        """
        Map a numpy array representing a solution vector to the internal list of variables.

        Parameters
        ----------
        solution : ArrayLike
            Numpy array representing the solution vector to be injected.

        Raises
        ------
        ValueError
            If the solution is not a 1D array or contains NaN values.
        ArrayLengthError
            If the solution length doesn't match the number of variables.
        """
        if len(solution.shape) != 1:
            raise ValueError("Solution must be a 1D array.")

        self._blocks_sa = {
            block_id: {service: 0 for service in self._blocks_cap[block_id].keys()}
            for block_id in self._blocks_lu.keys()
        }
        self._blocks_bfa_cp = copy.deepcopy(self._blocks_bfa)
        self._blocks_cap_cp = copy.deepcopy(self._blocks_cap)
        for var, val in zip(self._areas_X, solution):
            var.count = int(val)
            self._blocks_sa[var.block_id][var.service_type] += var.total_site_area

        if self._X_prev is not None:
            self._X = copy.deepcopy(self._X_prev)
            # Remove busy area, capacity and build floor area from previous solution
            for var in self._X:
                self._blocks_sa[var.block_id][var.service_type] -= var.total_site_area
                if self._blocks_lu[var.block_id] == LandUse.RESIDENTIAL:
                    self._blocks_bfa_cp[var.block_id] = 0
                else:
                    self._blocks_bfa_cp[var.block_id] -= var.total_build_floor_area
                self._blocks_cap_cp[var.block_id][var.service_type] -= var.total_capacity
        else:
            # Clear all area and capacity of solution
            for var in self._X:
                var.count = 0

        for var, val in zip(self._areas_X, solution):
            self._inject_area_to_X(var)
            var.count -= self._blocks_sa[var.block_id][var.service_type]

        self._X_prev_tmp = copy.deepcopy(self._X)
