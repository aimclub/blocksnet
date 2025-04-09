from typing import Dict, Set

from blocksnet.enums import LandUse

from .variable_adapter import VariableAdapter


class BlockSolution(VariableAdapter):
    """
    Class for adapting a solution vector specifically for block-level data in the city.
    """

    def __init__(self, blocks_lu: Dict[int, LandUse], service_types: Set[str]):
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

        # Get user-defined block land uses
        for block_id, land_use in blocks_lu.items():
            self._fill_X_with_block_variables(block_id, land_use, service_types)
