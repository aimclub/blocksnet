import copy
from typing import Dict

import numpy as np
from numpy.typing import ArrayLike

from blocksnet.method.annealing_optimizer import Indicator, LandUse
from blocksnet.models import City

from .area_checker import AreaChecker
from .provision_adapter import ProvisionAdapter
from .variable_adapter import VariableAdapter


class BlocksnetFacade:
    """
    The BlocksnetFacade class serves as a high-level interface for managing various 
    domain-related operations in an urban planning context. It integrates 
    variable adaptation, provision calculations, and area validation.

    Responsibilities
    ----------------
    - Manages land use data and associated indicators.
    - Provides an interface for computing urban planning metrics.
    - Handles transformations between raw input variables and domain-specific representations.
    - Computes distances for urban layout optimization.

    Attributes
    ----------
    _city : City
        The city model representing the urban layout.
    _blocks_lu : Dict[int, LandUse]
        A dictionary mapping block IDs to their respective land use categories.
    _area_checker : AreaChecker
        Utility for evaluating constraints and compliance with area-based conditions.
    _provision_adapter : ProvisionAdapter
        Adapter responsible for recalculating service provisions based on solution vector.
    _converter : VariableAdapter
        Adapter for converting optimization variables into domain-specific structures.
    _indicators : Dict[int, Indicator]
        A mapping of block IDs to their respective indicators, containing urban planning metrics.
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
            The city model object represents a block-network city information model.
        variable_adapter : VariableAdapter
            Adapter for converting solution variables into domain-specific data.
        """
        self._city: City = city_model
        self._blocks_lu: Dict[int, LandUse] = blocks_lu
        self._area_checker: AreaChecker = AreaChecker()
        self._provision_adapter: ProvisionAdapter = ProvisionAdapter(city_model, blocks_lu)
        self._converter: VariableAdapter = variable_adapter
        self._indicators = self._generate_indicators(blocks_lu, blocks_fsi, blocks_gsi)

    def _generate_indicators(
        self, blocks: Dict[int, LandUse], fsis: Dict[int, float], gsis: Dict[int, float]
    ) -> Dict[int, Indicator]:
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
        return {b_id: Indicator(self._city[b_id], blocks[b_id], fsis[b_id], gsis[b_id]) for b_id in blocks.keys()}

    def get_distance(self, x: ArrayLike) -> float:
        """
        Calculates the distance for the solution.

        Parameters
        ----------
        x : ArrayLike
            The input values representing the state of the blocks.

        Returns
        -------
        float
            The calculated distance.
        """
        X = self._converter(x)
        return self._area_checker.get_distance_for_entire_solution(X)

    def get_max_distance(self) -> float:
        """
        Retrieves the maximum possible distance for the current block setup.

        Returns
        -------
        float
            The maximum distance.
        """
        X = copy.deepcopy(self._converter.X)
        for i, x in enumerate(X):
            X[i].value = self.get_upper_bound_var(i)
        return self._area_checker.get_distance_for_entire_solution(X)

    def get_provisions(self, x: ArrayLike) -> Dict[str, float]:
        X = self._converter(x)
        return self._provision_adapter.recalculate_all(X, self._indicators)

    def get_changed_services_count(self, x: ArrayLike) -> int:
        X = self._converter(x)
        return len({var.service_type.name for var in X})

    def get_upper_bound_var(self, var_num: int) -> int:
        """
        Retrieves the upper bound for the variable at the given position.

        Parameters
        ----------
        var_pos : int
            The variable position.

        Returns
        -------
        int
            The upper bound for the variable.
        """
        var = self._converter.X[var_num]
        block_area = self._indicators[var.block.id].integrated_area
        return np.floor(block_area / var.brick.area)

    def get_var_weight(self, var_num: int) -> int:
        var = self._converter.X[var_num]
        return var.brick.area

    def get_limit(self, block_id: int) -> float:
        return self._indicators[block_id].integrated_area

    def get_group_num(self, var_num: int) -> int:
        return self._converter.X[var_num].block.id
