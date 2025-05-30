from abc import ABC
from typing import Dict, List, Optional, Set

import numpy as np
from numpy.typing import ArrayLike

from blocksnet import City, ServiceType
from blocksnet.method.annealing_optimizer import LandUse, Variable


class VariableAdapter(ABC):
    """
    Abstract base class for adapting a solution vector (numpy array) to city-level data,
    allowing interaction with city blocks, land use, and service types.
    This class handles the solution vector and maps it to the corresponding city variables.
    """

    class ArrayLengthError(Exception):
        """
        Exception raised when the length of the solution vector doesn't match the expected length.
        """

        def __init__(self, expected_length, actual_length):
            self.expected_length = expected_length
            self.actual_length = actual_length
            super().__init__(f"Expected solution length {expected_length}, but got {actual_length}.")

    def __init__(self) -> None:
        """
        Initializes the VariableAdapter instance with an empty list of variables.
        """
        self._X: List[Variable] = []

    @property
    def X(self) -> List[Variable]:
        """
        Returns the list of variables representing the current city-level data.

        Returns
        -------
        List[Variable]
            The list of variables representing the current solution.
        """
        return self._X

    def _fill_X_with_block_variables(
        self, city: City, block_id: int, land_use: LandUse, service_types: Optional[Set[ServiceType]] = None
    ):
        """
        Fills the solution vector with variables corresponding to a block, land use,
        and associated service types from the city model.

        Parameters
        ----------
        city : City
            The city model instance that provides blocks and service types.
        block_id : int
            The ID of the block to be processed.
        land_use : LandUse
            The land use type for the given block.
        service_types : Optional[Set[ServiceType]], optional
            The set of service types to consider for the given block, by default None.
        """

        chosen_service_types = set(city.get_land_use_service_types(land_use))

        if service_types is not None:
            chosen_service_types = chosen_service_types & {city[st.name] for st in service_types}

        for service_type in chosen_service_types:
            for brick in service_type.bricks:
                x = Variable(block=city[block_id], service_type=service_type, brick=brick)
                self._X.append(x)

    def _inject_solution_to_X(self, solution: ArrayLike):
        """
        Maps a numpy array representing a solution vector to the internal list of variables.

        Parameters
        ----------
        solution : ArrayLike
            Numpy array representing the solution vector to be injected.
        """
        if len(solution.shape) != 1:
            raise ValueError("Solution must be a 1D array.")

        if len(solution) != len(self._X):
            raise self.ArrayLengthError(len(self._X), len(solution))

        if np.any(np.isnan(solution)):  # Check NaNs
            raise ValueError("Solution contains NaN values.")

        for var, val in zip(self._X, solution):
            var.value = val

    def __call__(self, solution: ArrayLike) -> List[Variable]:
        """
        Makes the adapter callable, mapping a solution vector to the variables.

        Parameters
        ----------
        solution : ArrayLike
            The solution vector to map to the city variables.

        Returns
        -------
        List[Variable]
            The list of variables after injecting the solution vector.
        """
        self._inject_solution_to_X(solution)
        return self._X

    def __len__(self):
        """
        Returns the number of variables in the City-level solution List of Variables(_X).

        Returns
        -------
        int
            The number of variables in the solution.
        """
        return len(self._X)


class BlockSolution(VariableAdapter):
    """
    Class for adapting a solution vector specifically for block-level data in the city.
    """

    def __init__(self, city: City, blocks_lu: Dict[int, LandUse], service_types: Optional[Set[ServiceType]] = None):
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
            self._fill_X_with_block_variables(city, block_id, land_use, service_types)


class ServiceSolution(VariableAdapter):
    """
    Class for adapting a solution vector specifically for pool services in the city.
    It uses the city model and the selected service types to generate the corresponding variables.
    """

    def __init__(self, city: City, service_types: Set[ServiceType]):
        """
        Initializes the ServiceSolution adapter with the city and selected service types.

        Parameters
        ----------
        city : City
            The city model instance that provides the blocks and services.
        service_types : Set[ServiceType]
            The set of service types for which to generate variables.
        """
        super().__init__()

        # Get blocks land uses from the city model
        for block in city.blocks:
            if block.land_use is None:
                continue
            # If the block has no chosen services, _fill_X_with_block_variables will add nothing to X, which is acceptable
            self._fill_X_with_block_variables(city, block.id, block.land_use, service_types)
