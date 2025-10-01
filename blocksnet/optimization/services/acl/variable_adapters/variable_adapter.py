import copy
from abc import ABC
from typing import Dict, List

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
        self._blocks_bfa = {}
        self._blocks_cap = {}
        self._blocks_sa = {}
        self._X_prev = None
        self._X_prev_tmp = None

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
        Get the list of variables representing the current solution.

        Returns
        -------
        List[Variable]
            The list of variables in the adapter.
        """
        pass

    def add_service_type_vars(self, service_type: str):
        """
        Add variables for a specific service type to the adapter.

        Parameters
        ----------
        service_type : str
            The service type identifier to add variables for.
        """
        pass

    def add_variables(self, block_id: int, land_use: LandUse, service_type: str):
        """
        Add variables for a specific block, land use, and service type.

        Parameters
        ----------
        block_id : int
            ID of the block to add variables for.
        land_use : LandUse
            Land use type of the block.
        service_type : str
            Service type identifier to add variables for.
        """
        pass

    def _inject_solution_to_X(self, solution: ArrayLike):
        """
        Internal method to inject a solution vector into the variable list.

        Parameters
        ----------
        solution : ArrayLike
            Numpy array representing the solution vector.
        """
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

    def update_bfa(self, blocks_bfa: Dict[int, float]) -> None:
        """
        Update the build floor area values for blocks.

        Parameters
        ----------
        blocks_bfa : Dict[int, float]
            Dictionary mapping block IDs to their build floor area values.
        """
        self._blocks_bfa = blocks_bfa

    def update_cap(self, blocks_cap: Dict[int, Dict[str, float]]) -> None:
        """
        Update the capacity values for blocks and service types.

        Parameters
        ----------
        blocks_cap : Dict[int, Dict[str, float]]
            Nested dictionary mapping block IDs to service types to their capacity values.
        """
        self._blocks_cap = blocks_cap
        self.update_x_prev()

    def update_x_prev(self) -> None:
        """
        Update the previous solution state based on current temporary state.
        """
        if self._X_prev is not None or self._X_prev_tmp is None:
            self._X_prev = None
        else:
            self._X_prev = copy.deepcopy(self._X_prev_tmp)
