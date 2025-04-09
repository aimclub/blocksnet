from abc import ABC
from typing import List, Optional, Set

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from blocksnet.config.service_types.config import service_types_config
from blocksnet.enums import LandUse
from blocksnet.optimization.services.common.variable import Variable


class VariableAdapter(ABC):
    """
    Abstract base class for adapting a solution vector (numpy array) to city-level data,
    allowing interaction with city blocks, land use, and service types.
    
    This class handles the solution vector and maps it to the corresponding city variables,
    providing methods for conversion between array representations and variable objects.
    """

    class ArrayLengthError(Exception):
        """
        Exception raised when the length of the solution vector doesn't match the expected length.

        Attributes
        ----------
        expected_length : int
            The expected length of the solution vector.
        actual_length : int
            The actual length of the solution vector.
        """

        def __init__(self, expected_length, actual_length):
            """
            Initialize the ArrayLengthError with expected and actual lengths.

            Parameters
            ----------
            expected_length : int
                The expected length of the solution vector.
            actual_length : int
                The actual length of the solution vector.
            """
            self.expected_length = expected_length
            self.actual_length = actual_length
            super().__init__(f"Expected solution length {expected_length}, but got {actual_length}.")

    def __init__(self) -> None:
        """
        Initialize the VariableAdapter instance with an empty list of variables.
        """
        self._X: List[Variable] = []

    def _variables_to_df(self, variables: List[Variable]) -> pd.DataFrame:
        """
        Convert a list of Variable objects to a pandas DataFrame.

        Parameters
        ----------
        variables : List[Variable]
            List of Variable objects to convert.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the variable data.
        """
        data = [v.to_dict() for v in variables]
        return pd.DataFrame(data)

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

    def _fill_X_with_block_variables(self, block_id: int, land_use: LandUse, service_types: Optional[Set[str]] = None):
        """
        Populate the variables list with service variables for a specific block.

        Parameters
        ----------
        block_id : int
            ID of the block to process.
        land_use : LandUse
            Land use type of the block.
        service_types : Optional[Set[str]], optional
            Set of service types to include. If None, all valid services for the land use will be included.
        """
        chosen_service_types = set(service_types_config[land_use])

        if service_types is not None:
            chosen_service_types = chosen_service_types & service_types

        units = service_types_config.units

        for _, unit in units[units.service_type.isin(chosen_service_types)].iterrows():
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

        if len(solution) != len(self._X):
            raise self.ArrayLengthError(len(self._X), len(solution))

        if np.any(np.isnan(solution)):  # Check NaNs
            raise ValueError("Solution contains NaN values.")

        for var, val in zip(self._X, solution):
            var.count = val

    def __call__(self, solution: ArrayLike) -> List[Variable]:
        """
        Convert a solution vector to a list of Variable objects.

        Parameters
        ----------
        solution : ArrayLike
            Numpy array representing the solution vector.

        Returns
        -------
        List[Variable]
            List of Variable objects representing the solution.
        """
        self._inject_solution_to_X(solution)
        return self._X

    def __len__(self) -> int:
        """
        Get the number of variables in the adapter.

        Returns
        -------
        int
            The number of variables.
        """
        return len(self._X)