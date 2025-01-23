from typing import Dict

from blocksnet.method.annealing_optimizer import Indicator, LandUse, Variable
from blocksnet.models import City

from .provision_adapter import ProvisionAdapter
from .variable_adapter import VariableAdapter
from .constraints_checker import Constraints


class BlocksnetFacade:
    """
    Facade class for managing domain-related operations, including provision calculations.
    """

    def __init__(
        self,
        blocks_lu: Dict[int, LandUse],
        blocks_fsi: Dict[int, float],
        blocks_gsi: Dict[int, float],
        city_model: City,
        variable_adapter: VariableAdapter,
    ) -> None:
        """
        Initializes the BlocksnetFacade class with the necessary parameters.

        Parameters
        ----------
        blocks_lu : Dict[int, LandUse]
            Dictionary mapping block IDs to their corresponding land uses.
        blocks_fsi : Dict[int, float]
            Dictionary mapping block IDs to their Floor Space Index (FSI) values.
        blocks_gsi : Dict[int, float]
            Dictionary mapping block IDs to their Ground Space Index (GSI) values.
        city_model : City
            The city model object representing the urban layout.
        variable_adapter : VariableAdapter
            Adapter for converting solution variables into domain-specific data.
        """
        self.city: City = city_model
        self.blocks_lu: Dict[int, LandUse] = blocks_lu
        self.provision_adapter: ProvisionAdapter = ProvisionAdapter(city_model, blocks_lu, blocks_gsi)
        self.constraints_checker: Constraints = #TODO
        self.converter: VariableAdapter = variable_adapter

        # Generate indicators for the blocks
        self.indicators: Dict[int, Indicator] = self._generate_indicators(
            blocks_lu, blocks_fsi, blocks_gsi
        )

    def _generate_indicators(self, blocks: Dict[int, LandUse], fsis: Dict[int, float], gsis: Dict[int, float]) -> Dict[int, Indicator]:
        """
        Generates indicators for each block based on the provided data.

        Parameters
        ----------
        blocks : dict[int, LandUse]
            Dictionary mapping block IDs to their corresponding land uses.
        fsis : dict[int, float]
            Dictionary mapping block IDs to their Floor Space Index (FSI) values.
        gsis : dict[int, float]
            Dictionary mapping block IDs to their Ground Space Index (GSI) values.

        Returns
        -------
        dict[int, Indicator]
            A dictionary mapping block IDs to their corresponding indicators.
        """
        return {
            b_id: Indicator(self.city[b_id], blocks[b_id], fsis[b_id], gsis[b_id])
            for b_id in blocks.keys()
        }
    pass