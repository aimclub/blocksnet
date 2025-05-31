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
    The penalty increases exponentially with the distance beyond a maximum threshold.
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
        self._lambda = 0.1  # Penalty coefficient

    def _calc(self, x: ArrayLike) -> float:
        """
        Calculate the penalty based on the distance between the current solution
        and the maximum allowed distance.

        The penalty grows exponentially as the distance exceeds the maximum allowed distance.

        Parameters
        ----------
        x : ArrayLike
            Array representing the solution.

        Returns
        -------
        float
            The calculated penalty value.
        """
        dist: float = self._facade.get_distance(x)
        max_dist: float = self._facade.get_max_distance()
        return np.exp(dist - max_dist) * self._lambda


class Objective(ABC):
    """
    Abstract base class defining the objective function for optimization problems.
    Handles function evaluation counting and penalty calculations.
    """

    def __init__(self, num_params: int, facade: Facade, max_evals: Optional[int], penalty_func: Optional[Penalty]):
        """
        Initialize the objective function with parameters and evaluation limits.

        Parameters
        ----------
        num_params : int
            Number of parameters in the optimization problem.
        facade : Facade
            The facade providing access to data related to the optimization problem.
        max_evals : Optional[int]
            Maximum number of function evaluations allowed (None for unlimited).
        penalty_func : Optional[Penalty]
            The penalty function used in the objective function (defaults to DistancePenalty if None).
        """
        self._num_params: int = num_params
        self._current_func_evals: int = 0
        self._max_func_evals: Optional[int] = max_evals
        self._facade: Facade = facade
        self._penalty: Penalty = DistancePenalty(facade) if penalty_func is None else penalty_func
        self._x_last = np.zeros(num_params)  # Stores the last evaluated solution

    def __call__(self, x: ArrayLike) -> tuple[Dict[str, float], float]:
        """
        Evaluate the objective function for a given solution by calling evaluate().

        Parameters
        ----------
        x : ArrayLike
            Array representing the solution.

        Returns
        -------
        tuple[Dict[str, float], float]
            A tuple containing:
            - Dictionary of provision values for each service
            - The computed objective function value
        """
        return self.evaluate(x)

    @abstractmethod
    def evaluate(self, x: ArrayLike) -> tuple[Dict[str, float], float]:
        """
        Abstract method to evaluate the objective function for a given solution.
        Must be implemented by concrete subclasses.

        Parameters
        ----------
        x : ArrayLike
            Array representing the solution.

        Returns
        -------
        tuple[Dict[str, float], float]
            A tuple containing:
            - Dictionary of provision values for each service
            - The computed objective function value
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
        Get the current number of function evaluations performed.

        Returns
        -------
        int
            The current number of function evaluations.
        """
        return self._current_func_evals

    def get_penalty(self, x: ArrayLike) -> float:
        """
        Calculate the penalty for the current solution using the configured penalty function.

        Parameters
        ----------
        x : ArrayLike
            Array representing the solution.

        Returns
        -------
        float
            The computed penalty value.
        """
        return self._penalty(x)

    def check_available_evals(self) -> bool:
        """
        Check if more function evaluations can be performed within the limit.

        Returns
        -------
        bool
            True if more evaluations can be performed (or if max_evals is None),
            False if the evaluation limit has been reached.
        """
        if self._max_func_evals is None:
            return True
        return self._current_func_evals < self._max_func_evals

    def check_optimize_need(self) -> bool:
        """
        Check if optimization is needed by comparing current provisions to ideal values.

        Returns
        -------
        bool
            True if any service provision differs significantly from 1.0,
            indicating optimization is needed.
        """
        return any(abs(value - 1.0) > 1e-8 for value in self._facade.last_provisions.values())


class ThresholdObjective(Objective):
    """
    Implementation of an objective function that evaluates solutions based on whether
    they meet certain thresholds for service provisions.
    """

    def __init__(self, num_params: int, facade: Facade, max_evals: int, penalty_func: Optional[Penalty] = None):
        """
        Initialize the threshold objective function.

        Parameters
        ----------
        num_params : int
            Number of parameters in the problem.
        facade : Facade
            The facade providing access to data.
        max_evals : int
            Maximum number of function evaluations.
        penalty_func : Optional[Penalty]
            The penalty function to use (defaults to DistancePenalty if None).
        """
        super().__init__(num_params, facade, max_evals, penalty_func)

    def evaluate(self, x: ArrayLike) -> tuple[Dict[str, float], float]:
        """
        Evaluate the objective function for a given solution based on threshold criteria.

        Note: This is currently a placeholder implementation that calls the parent class method.
        Future implementation will evaluate solutions based on whether they meet certain
        thresholds for service provisions.

        Parameters
        ----------
        x : ArrayLike
            Array representing the solution.

        Returns
        -------
        tuple[Dict[str, float], float]
            A tuple containing:
            - Dictionary of provision values for each service
            - The computed objective function value
        """
        return super().evaluate(x)


class WeightedObjective(Objective):
    """
    Implementation of an objective function that uses weighted sums of service provisions
    to evaluate solutions.
    """

    def __init__(
        self,
        num_params: int,
        facade: Facade,
        max_evals: int,
        weights: Dict[str, float],
        penalty_func: Optional[Penalty] = None,
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
            Dictionary mapping service names to their respective weights.
        penalty_func : Optional[Penalty]
            The penalty function to use (defaults to DistancePenalty if None).
        """
        super().__init__(num_params, facade, max_evals, penalty_func)
        self._weights = weights  # Dictionary of service weights

    def evaluate(self, x: ArrayLike) -> tuple[Dict[str, float], float]:
        """
        Evaluate the weighted objective function for a given solution.

        The objective value is computed as a weighted sum of service provisions,
        where the weights are specified during initialization.

        Parameters
        ----------
        x : ArrayLike
            Array representing the solution.

        Returns
        -------
        tuple[Dict[str, float], float]
            A tuple containing:
            - Dictionary of provision values for each service
            - The computed weighted sum objective value
        """
        # Get provisions for the solution
        if np.count_nonzero(x) == 0:
            provisions = self._facade.start_provisions
        else:
            provisions = self._facade.get_provisions(self._x_last, x)
            changed_services = self._facade.get_changed_services(self._x_last, x)
            self._current_func_evals += len(changed_services)

        self._x_last = x  # Store current solution for next evaluation

        # Calculate weighted sum of provisions for services with defined weights
        services = self._weights.keys() & provisions.keys()
        obj_value = sum(provisions[st] * self._weights[st] for st in services)

        return provisions, obj_value
