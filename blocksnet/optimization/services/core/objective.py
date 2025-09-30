from abc import ABC, abstractmethod
from typing import Dict, Optional

import numpy as np
from numpy.typing import ArrayLike

from blocksnet.optimization.services.acl import Facade


class Penalty(ABC):
    """
    Abstract base class representing a penalty function for an optimization problem.
    """

    def __init__(self):
        """Initialize the instance.

        Returns
        -------
        None
            Description.

        """
        pass

    @abstractmethod
    def _calc(self, x: ArrayLike) -> float:
        """
        Calculate the penalty for a given solution.

        Parameters
        ----------
        x : ArrayLike
            Array representing the solution.

        Returns
        -------
        float
            The calculated penalty.
        """
        pass

    def __call__(self, x: ArrayLike) -> float:
        """
        Calculate the penalty by calling the _calc method.

        Parameters
        ----------
        x : ArrayLike
            Array representing the solution.

        Returns
        -------
        float
            The calculated penalty.
        """
        return self._calc(x)


class DistancePenalty(Penalty):
    """
    Implementation of a distance-based penalty function used in optimization problems.
    """

    def __init__(self, facade: Facade):
        """
        Initialize the distance penalty function with the given facade for data access.

        Parameters
        ----------
        facade : Facade
            The facade providing access to data related to the optimization problem.
        """
        self._facade: Facade = facade
        self._lambda = 0.1

    def _calc(self, x: ArrayLike) -> float:
        """
        Calculate the penalty based on the distance between the current solution and the maximum distance.

        Parameters
        ----------
        x : ArrayLike
            Array representing the solution.

        Returns
        -------
        float
            The calculated penalty.
        """
        dist: float = self._facade.get_distance(x)
        max_dist: float = self._facade.get_max_distance()
        return np.exp(dist - max_dist) * self._lambda


class Objective(ABC):
    """
    Abstract base class defining the objective function for optimization problems.
    """

    def __init__(self, num_params: int, facade: Facade, max_evals: Optional[int], penalty_func: Optional[Penalty]):
        """
        Initialize the objective function.

        Parameters
        ----------
        num_params : int
            Number of parameters in the optimization problem.
        facade : Facade
            The facade providing access to data related to the optimization problem.
        max_evals : Optional[int]
            Maximum number of function evaluations allowed.
        penalty_func : Optional[Penalty]
            The penalty function used in the objective function.
        """
        self._num_params: int = num_params
        self._current_func_evals: int = 0
        self._max_func_evals: Optional[int] = max_evals
        self._facade: Facade = facade
        self._penalty: Penalty = DistancePenalty(facade) if penalty_func is None else penalty_func
        self._x_last = np.zeros(num_params)

    def __call__(self, x: ArrayLike, suggested: bool = True) -> tuple[Dict[str, float], float]:
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
        return self.evaluate(x, suggested)

    @abstractmethod
    def evaluate(self, x: ArrayLike, suggested: bool = True) -> tuple[Dict[str, float], float]:
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

    def get_penalty(self, x: ArrayLike) -> float:
        """
        Get the penalty for the current solution.

        Parameters
        ----------
        x : ArrayLike
            Array representing the solution.

        Returns
        -------
        float
            The penalty for the current solution.
        """
        return self._penalty(x)

    def check_available_evals(self):
        """
        Check if more function evaluations can be performed.

        Returns
        -------
        bool
            True if more evaluations can be performed, otherwise False.
        """
        return self._current_func_evals < self._max_func_evals

    def check_optimize_need(self):
        """Check optimize need.

        """
        return any(abs(value - 1.0) > 1e-8 for value in self._facade.last_provisions.values())


class WeightedObjective(Objective):
    """
    Implementation of an objective function with weights for different parameters in the optimization problem.
    """

    def __init__(
        self,
        num_params: int,
        facade: Facade,
        max_evals: int,
        weights: Dict[str, float],
        penalty_func: Penalty = None,
    ):
        """
        Initialize the weighted objective function.

        Parameters
        ----------
        num_params : int
            Number of parameters in the problem.
        facade : Facade
            The facade providing access to data.
        max_evals : int
            Maximum number of function evaluations.
        weights : Dict[str, float]
            A dictionary containing weights for the different parameters in the problem.
        penalty_func : Penalty
            The penalty function.
        """
        super().__init__(num_params, facade, max_evals, penalty_func)
        self._weights = weights

    def evaluate(self, x: ArrayLike, suggested: bool = True) -> tuple[Dict[str, float], float]:
        """
        Evaluate the weighted objective function for a given solution.

        Parameters
        ----------
        x : ArrayLike
            Array representing the solution.

        suggested : bool, optional
            Whether the solution is suggested or not. If False, it will not update the last solution

        Returns
        -------
        tuple[Dict[str, float], float]
            A tuple containing a dictionary of values and the total value of the objective function.
        """

        if np.count_nonzero(x) == 0:
            provisions = self._facade.start_provisions
        else:
            provisions, changed_services = self._facade.get_provisions(self._x_last, x)

            if not suggested:
                changed_services = {}
            self._current_func_evals += len(changed_services)

            self._x_last = np.array([var.count for var in self._facade.get_X(x)])

        services = self._weights.keys() & provisions.keys()
        obj_value = sum(provisions[st] * self._weights[st] for st in services)
        return provisions, obj_value
