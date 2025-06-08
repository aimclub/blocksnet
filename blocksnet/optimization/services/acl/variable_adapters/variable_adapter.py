from abc import ABC, abstractmethod
from typing import List

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from blocksnet.enums import LandUse
from blocksnet.optimization.services.common.variable import Variable


class VariableAdapter(ABC):
    """
    Abstract base class for adapting solution vectors to service allocation variables.

    Provides the interface for converting between numerical solution vectors and
    Variable objects representing service allocations across city blocks.
    """

    class ArrayLengthError(Exception):
        """
        Exception raised when a solution vector has incorrect dimensions.

        Attributes
        ----------
        expected_length : int
            The expected number of variables in the solution.
        actual_length : int
            The actual number of elements in the solution vector.
        message : str
            Human-readable error message explaining the mismatch.
        """

        def __init__(self, expected_length: int, actual_length: int):
            """
            Initialize the ArrayLengthError with dimension information.

            Parameters
            ----------
            expected_length : int
                Number of variables expected in the solution.
            actual_length : int
                Number of elements found in the solution vector.
            """
            self.expected_length = expected_length
            self.actual_length = actual_length
            super().__init__(f"Expected solution length {expected_length}, but got {actual_length}.")

    def __init__(self) -> None:
        """
        Initialize the VariableAdapter with empty variable storage.
        """
        self._X: List[Variable] = []

    def _variables_to_df(self, variables: List[Variable]) -> pd.DataFrame:
        """
        Convert a list of Variable objects to a structured DataFrame.

        Parameters
        ----------
        variables : List[Variable]
            List of Variable objects to convert. Each variable should have
            a to_dict() method that returns its attributes as a dictionary.

        Returns
        -------
        pd.DataFrame
            DataFrame where each row represents a variable and columns
            represent variable attributes (block_id, service_type, capacity, etc.)
        """
        data = [v.to_dict() for v in variables]
        return pd.DataFrame(data)

    @property
    @abstractmethod
    def X(self) -> List[Variable]:
        """
        Abstract property representing the current list of variables.

        Returns
        -------
        List[Variable]
            The current collection of service allocation variables.
        """
        pass

    @abstractmethod
    def add_service_type_vars(self, service_type: str):
        """
        Abstract method to add variables for a specific service type.

        Parameters
        ----------
        service_type : str
            Identifier of the service type to add variables for.
        """
        pass

    @abstractmethod
    def add_variables(self, block_id: int, land_use: LandUse, service_type: str):
        """
        Abstract method to add variables for a specific block and service type.

        Parameters
        ----------
        block_id : int
            Identifier of the city block.
        land_use : LandUse
            Land use designation of the block.
        service_type : str
            Service type to allocate on this block.
        """
        pass

    @abstractmethod
    def _inject_solution_to_X(self, solution: ArrayLike):
        """
        Abstract method to map a solution vector to internal variables.

        Parameters
        ----------
        solution : ArrayLike
            1D numpy array where each element corresponds to a variable count.
        """
        pass

    def __call__(self, solution: ArrayLike) -> List[Variable]:
        """
        Convert a solution vector to Variable objects and return them.

        Parameters
        ----------
        solution : ArrayLike
            Numerical vector representing service unit counts.

        Returns
        -------
        List[Variable]
            List of Variable objects populated with solution values.

        Note
        ----
        Internally calls _inject_solution_to_X() to perform the conversion.
        """
        self._inject_solution_to_X(solution)
        return self._X

    def __len__(self) -> int:
        """
        Get the number of variables currently managed by this adapter.

        Returns
        -------
        int
            Count of variables in the adapter's collection.
        """
        return len(self.X)
