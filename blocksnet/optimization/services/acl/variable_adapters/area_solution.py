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


def unit_area(unit) -> float:
    """
    Area a single service unit consumes from the area granted by the solution.

    A service either takes a plot of its own (``site_area > 0``) or is built
    into a building and only takes build floor area. Both are metered against
    the same granted area: otherwise built-in services cost the solution
    nothing and are placed up to the block's whole floor area budget no matter
    what the solution vector says.

    Parameters
    ----------
    unit : Unit | Variable
        Object exposing ``site_area`` and ``build_floor_area``.

    Returns
    -------
    float
        Area consumed by one unit, ``0`` if the unit consumes no area at all.
    """
    return unit.site_area if unit.site_area > 0 else unit.build_floor_area


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

        # One unit of this variable is one square metre granted to the service
        # type in the block. It is charged to both area budgets of the block,
        # since the units it expands into may consume either of them.
        x_area = AreaVariable(block_id=block_id, service_type=service_type, site_area=1, build_floor_area=1, capacity=0)

        units = service_types_config.units

        units = units[units.service_type == service_type]
        self._units_lists.update({service_type: []})

        for _, unit in units.iterrows():
            x = Variable(block_id=block_id, **unit)
            u = Unit(site_area=x.site_area, build_floor_area=x.build_floor_area, capacity=x.capacity)
            self._units_lists[service_type].append(u)
            self._X.append(x)

        def capacity_sortkey(u: Unit):
            # Largest facility first, so a service is delivered by as few of
            # them as possible; ties go to the one taking the least area.
            # Ranking by capacity per square metre instead made a built-in
            # school (2400 m2 of floor for 250 places) beat a real one with
            # grounds (33000 m2 of plot for 1100), because it treated plot area
            # and floor area as the same resource. They are not.
            return (-u.capacity, unit_area(u))

        self._units_lists[service_type].sort(key=capacity_sortkey)

        # Capacity produced per square metre of granted area, taken from the
        # unit the injection reaches for first. It has to match the ordering
        # above: with the densest unit instead, the granted area was bounded by
        # `demand / density_of_the_smallest_built_in_unit`, which for a school
        # never reached the 33000 m2 a real one needs, so a facility with
        # grounds could not be granted enough area to ever be placed.
        leading = next((u for u in self._units_lists[service_type] if unit_area(u) > 0), None)
        if leading is not None:
            x_area.capacity = leading.capacity / unit_area(leading)
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
                    x_unit_area = unit_area(x)
                    addition = 1e9
                    if x.capacity != 0:
                        # Demand belongs to a catchment, not to a block. Taking
                        # only this block's budget let every block of a shared
                        # catchment serve that catchment in full: three blocks
                        # placed 690 library places against a demand of 249.
                        dependents = [
                            d
                            for d in self.cap_dependents(x.block_id, x.service_type)
                            if x.service_type in self._blocks_cap_cp.get(d, {})
                        ]
                        available = min(
                            (self._blocks_cap_cp[d][x.service_type] for d in dependents),
                            default=self._blocks_cap_cp[x.block_id][x.service_type],
                        )
                        addition = int(np.floor(available / x.capacity))
                    if x.build_floor_area != 0:
                        addition = min(addition, int(np.floor(self._blocks_bfa_cp[x.block_id] / x.build_floor_area)))
                        if x_unit_area == x.build_floor_area:  # built-in, so it sits on the ground floor
                            ground_floor = self._blocks_gfa_cp.get(x.block_id, float("inf"))
                            addition = min(addition, int(np.floor(ground_floor / x.build_floor_area)))
                    # Every unit is charged to the granted area, whether it
                    # takes a plot or is built in. A unit consuming no area at
                    # all cannot be metered and is never placed.
                    if x_unit_area == 0:
                        addition = 0
                    else:
                        addition = min(
                            addition, int(np.floor(self._blocks_sa[x.block_id][x.service_type] / x_unit_area))
                        )
                    addition = max(addition, 0)
                    x.count += addition
                    self._blocks_sa[x.block_id][x.service_type] -= x_unit_area * addition
                    self._blocks_bfa_cp[x.block_id] -= x.build_floor_area * addition
                    if x.site_area == 0 and x.block_id in self._blocks_gfa_cp:
                        self._blocks_gfa_cp[x.block_id] -= x.build_floor_area * addition
                    for dependent in self.cap_dependents(x.block_id, x.service_type):
                        if x.service_type in self._blocks_cap_cp.get(dependent, {}):
                            self._blocks_cap_cp[dependent][x.service_type] -= x.capacity * addition
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
        self._blocks_gfa_cp = copy.deepcopy(self._blocks_gfa)
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
                if var.site_area == 0 and var.block_id in self._blocks_gfa_cp:
                    self._blocks_gfa_cp[var.block_id] -= var.total_build_floor_area
                for dependent in self.cap_dependents(var.block_id, var.service_type):
                    if var.service_type in self._blocks_cap_cp.get(dependent, {}):
                        self._blocks_cap_cp[dependent][var.service_type] -= var.total_capacity
        else:
            # Clear all area and capacity of solution
            for var in self._X:
                var.count = 0

        for var, val in zip(self._areas_X, solution):
            self._inject_area_to_X(var)
            var.count -= self._blocks_sa[var.block_id][var.service_type]

        self._X_prev_tmp = copy.deepcopy(self._X)
