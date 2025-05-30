from typing import Dict, List

import numpy as np
from numpy.typing import ArrayLike

from blocksnet.config.service_types.config import service_types_config
from blocksnet.enums import LandUse
from blocksnet.optimization.services.common.variable import Variable

from .variable_adapter import VariableAdapter


class BlockSolution(VariableAdapter):
    """
    Class for adapting a solution vector specifically for block-level data in the city.
    """

    def __init__(self, blocks_lu: Dict[int, LandUse]):
        """
        Initializes the BlockSolution adapter with the city and the land use data for blocks.

        Parameters
        ----------
        city : City
            The city model instance that provides the blocks and service types.
        blocks_lu : Dict[int, LandUse]
            A dictionary mapping block IDs to their corresponding land use types.
        service_types : Optional[Set[ServiceType]], optional
            The set of service types for the blocks, by default None.
        """
        super().__init__()
        self._blocks_lu: Dict[int, LandUse] = blocks_lu

    @property
    def X(self) -> List[Variable]:
        """
        Get the list of variables representing the current city-level data.

        Returns
        -------
        List[Variable]
            The list of variables representing the current solution.
        """
        return self._X

    def add_service_type_vars(self, service_type: str):
        for block_id, land_use in self._blocks_lu.items():
            self.add_variables(block_id, land_use, service_type)

    def add_variables(self, block_id: int, land_use: LandUse, service_type: str):
        if service_type not in service_types_config[land_use]:
            return

        units = service_types_config.units

        for _, unit in units[units.service_type == service_type].iterrows():
            x = Variable(block_id=block_id, **unit)
            self._X.append(x)

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

        for var, val in zip(self._X, solution):
            var.count = int(val)
