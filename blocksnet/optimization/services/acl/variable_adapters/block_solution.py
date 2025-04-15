from typing import Dict, Set

from blocksnet.enums import LandUse

from .variable_adapter import VariableAdapter


class BlockSolution(VariableAdapter):
    """
    Class for adapting a solution vector specifically for block-level data in the city.

    This adapter handles the conversion between optimization variables and block-level
    service allocation solutions, taking into account land use constraints.
    """

    def __init__(self, blocks_lu: Dict[int, LandUse]):
        """
        Initializes the BlockSolution adapter with the city and the land use data for blocks.

        Parameters
        ----------
        blocks_lu : Dict[int, LandUse]
            A dictionary mapping block IDs to their corresponding land use types.
        """
        super().__init__()
        self._blocks_lu: Dict[int, LandUse] = blocks_lu

    def add_service_type_vars(self, service_type: str):
        """
        Add variables for a specific service type to all applicable blocks based on their land use.

        Parameters
        ----------
        service_type : str
            The service type identifier for which variables should be created.
            This service type must be compatible with the land use types of the blocks.
        """
        for block_id, land_use in self._blocks_lu.items():
            self._fill_X_with_block_variables(block_id, land_use, {service_type})