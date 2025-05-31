from typing import Dict, List

import numpy as np
from numpy.typing import ArrayLike

from blocksnet.config.service_types.config import service_types_config
from blocksnet.enums import LandUse
from blocksnet.optimization.services.common.variable import Variable

from .variable_adapter import VariableAdapter


class BlockSolution(VariableAdapter):
    """
    A class for managing block-level service allocation solutions.

    This adapter handles the conversion between solution vectors and block-level service
    variables, providing methods to add service variables and inject solutions.
    """

    def __init__(self, blocks_lu: Dict[int, LandUse]):
        """
        Initialize the BlockSolution with land use data for city blocks.

        Parameters
        ----------
        blocks_lu : Dict[int, LandUse]
            Dictionary mapping block IDs to their land use types.
            Keys are block IDs (integers), values are LandUse enum values.
        """
        super().__init__()
        self._blocks_lu: Dict[int, LandUse] = blocks_lu

    @property
    def X(self) -> List[Variable]:
        """
        Get the list of variables representing the current solution.

        Returns
        -------
        List[Variable]
            List of Variable objects representing the current service allocation solution.
            Each variable contains block_id, service_type, and capacity information.
        """
        return self._X

    def add_service_type_vars(self, service_type: str):
        """
        Add variables for a specific service type across all blocks.

        Parameters
        ----------
        service_type : str
            The service type identifier to add variables for.
            Must match a service type defined in the service_types_config.
        """
        for block_id, land_use in self._blocks_lu.items():
            self.add_variables(block_id, land_use, service_type)

    def add_variables(self, block_id: int, land_use: LandUse, service_type: str):
        """
        Add service variables for a specific block and service type.

        Parameters
        ----------
        block_id : int
            The ID of the block to add variables for.
        land_use : LandUse
            The land use type of the block.
        service_type : str
            The service type identifier to add variables for.
            Must be compatible with the block's land use type.
        """
        if service_type not in service_types_config[land_use]:
            return

        # Get all unit configurations for this service type
        units = service_types_config.units
        service_units = units[units.service_type == service_type]

        # Create a variable for each unit configuration
        for _, unit in service_units.iterrows():
            x = Variable(block_id=block_id, **unit)
            self._X.append(x)

    def _inject_solution_to_X(self, solution: ArrayLike):
        """
        Map a solution vector to the internal list of variables.

        Parameters
        ----------
        solution : ArrayLike
            1D numpy array representing the solution vector.
            Each element corresponds to a variable in self._X and represents
            the count of units to allocate.

        Raises
        ------
        ValueError
            If the solution is not a 1D array.
            If the solution length doesn't match the number of variables.
        """
        if len(solution.shape) != 1:
            raise ValueError("Solution must be a 1D array.")

        # Update each variable's count with the corresponding solution value
        for var, val in zip(self._X, solution):
            var.count = int(val)
