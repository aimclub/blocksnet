from abc import ABC, abstractmethod
from typing import Dict, Optional

import numpy as np
from numpy.typing import ArrayLike

from blocksnet.optimizer.src.acl import BlocksnetFacade


class Objective(ABC):
    """
    Abstract base class defining the objective function for optimization problems.
    """

    def __init__(
        self, num_params: int, facade: BlocksnetFacade, max_evals: Optional[int]
    ):
        """
        Initialize the objective function.

        Parameters
        ----------
        num_params : int
            Number of parameters in the optimization problem.
        facade : BlocksnetFacade
            The facade providing access to data related to the optimization problem.
        max_evals : Optional[int]
            Maximum number of function evaluations allowed.
        """
        self._num_params: int = num_params
        self._current_func_evals: int = 0
        self._max_func_evals: Optional[int] = max_evals
        self._facade: BlocksnetFacade = facade

    def __call__(self, x: ArrayLike) -> tuple[Dict[str, float], float]:
        """
        Evaluate the objective function for a given solution.

        Parameters
        ----------
        x : ArrayLike
            Array representing the solution.

        Returns
        -------
        tuple[Dict[str, float], float]
            A tuple containing a dictionary of values and the value of the objective function.
        """
        return self.evaluate(x)

    @abstractmethod
    def evaluate(self, x: ArrayLike) -> tuple[Dict[str, float], float]:
        """
        Evaluate the objective function for a given solution.

        Parameters
        ----------
        x : ArrayLike
            Array representing the solution.

        Returns
        -------
        tuple[Dict[str, float], float]
            A tuple containing a dictionary of values and the value of the objective function.
        """
        pass

    @property
    def num_params(self) -> int:
        """
        Get the number of parameters in the optimization problem.

        Returns
        -------
        int
            The number of parameters.
        """
        return self._num_params

    @property
    def current_func_evals(self) -> int:
        """
        Get the current number of function evaluations.

        Returns
        -------
        int
            The current number of function evaluations.
        """
        return self._current_func_evals

    def check_available_evals(self):
        """
        Check if more function evaluations can be performed.

        Returns
        -------
        bool
            True if more evaluations can be performed, otherwise False.
        """
        return self._current_func_evals < self._max_func_evals


class WeightedObjective(Objective):
    """
    Implementation of an objective function with weights for different parameters in the optimization problem.
    """

    def __init__(
        self,
        num_params: int,
        facade: BlocksnetFacade,
        max_evals: int,
        weights: Dict[str, float],
    ):
        """
        Initialize the weighted objective function.

        Parameters
        ----------
        num_params : int
            Number of parameters in the problem.
        facade : BlocksnetFacade
            The facade providing access to data.
        max_evals : int
            Maximum number of function evaluations.
        weights : Dict[str, float]
            A dictionary containing weights for the different parameters in the problem.
        """
        super().__init__(num_params, facade, max_evals)
        self._weights = weights

    def evaluate(self, x: ArrayLike) -> tuple[Dict[str, float], float]:
        """
        Evaluate the weighted objective function for a given solution.

        Parameters
        ----------
        x : ArrayLike
            Array representing the solution.

        Returns
        -------
        tuple[Dict[str, float], float]
            A tuple containing a dictionary of values and the total value of the objective function.
        """
        provisions = self._facade.get_provisions(x)
        self._current_func_evals += self._facade.get_changed_services_count(x)
        obj_value = sum(prov * self._weights[st] for st, prov in provisions.items())
        return provisions, obj_value


class PenaltyObjective(Objective):
    pass


class ThresholdObjective(Objective): # TODO: implement when get Thresholds for integral function
    """
    Implementation of an objective function with a threshold, used for optimization problems with constraints.
    """
    pass
