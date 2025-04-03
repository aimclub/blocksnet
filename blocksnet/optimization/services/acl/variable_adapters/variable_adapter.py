from abc import ABC
from typing import Dict, List

import numpy as np
from numpy.typing import ArrayLike

import pandas as pd

from blocksnet.enums import LandUse
from ...common import Variable


class VariableAdapter(ABC):

    class ArrayLengthError(Exception):
        """
        Exception raised when the length of the solution vector doesn't match the expected length.
        """

        def __init__(self, expected_length, actual_length):
            self.expected_length = expected_length
            self.actual_length = actual_length
            super().__init__(f"Expected solution length {expected_length}, but got {actual_length}.")

    def __init__(self) -> None:
       pass

    def _initialize_variables(self, blocks_lus: Dict[int, LandUse]) -> list[Variable]:
        pass
    
    def _variables_to_df(self, variables: list[Variable]) -> pd.DataFrame:
        data = [v.to_dict() for v in variables]
        return pd.DataFrame(data)

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
        self._inject_solution_to_X(solution)
        return self._X

    def __len__(self):
        return len(self._X)


