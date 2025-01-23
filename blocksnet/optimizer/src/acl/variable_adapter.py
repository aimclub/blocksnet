from abc import ABC, abstractmethod
from typing import List

from numpy.typing import ArrayLike

from blocksnet.method.annealing_optimizer import Variable


class VariableAdapter(ABC):
    """
    Abstract base class for adapting solution vector to city data.
    """

    def __init__(self, X: List[Variable]) -> None:
        self._X: List[Variable] = X

    @abstractmethod
    def _inject_solution_to_X(self, solution: ArrayLike):
        """
        Strategy for mapping numpy array to a list of variables.

        Parameters
        ----------
        solution : ArrayLike
            Numpy array representing the solution vector.
        """
        pass

    def __call__(self, solution: ArrayLike) -> List[Variable]:
        self._inject_solution_to_X(solution)
        return self.X

    @property
    def X(self) -> List[Variable]:
        """
        Returns the list of variables.

        Returns
        -------
        List[Variable]
            List of variables representing city-level data.
        """
        return self._X


class ServiceSolution(VariableAdapter):
    pass  # TODO: Implement service-specific logic


class CitySolution(VariableAdapter):
    pass  # TODO: Implement city-specific logic


class BlockSolution(VariableAdapter):
    pass  # TODO: Implement block-specific logic
