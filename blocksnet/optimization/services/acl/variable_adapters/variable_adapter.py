from abc import ABC
from typing import List

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

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
        pass

    def add_service_type_vars(self, service_type: str):
        pass

    def add_variables(self, block_id: int, land_use: LandUse, service_type: str):
        pass

    def _inject_solution_to_X(self, solution: ArrayLike):
        pass

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
        return len(self.X)
