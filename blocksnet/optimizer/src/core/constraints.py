from abc import ABC, abstractmethod
from typing import Callable, Dict

import numpy as np
from numpy.typing import ArrayLike

from blocksnet.optimizer.src.acl import BlocksNetFacade


class Constraints(ABC):
    """
    Abstract base class defining a set of constraints for optimization problems.
    """

    @abstractmethod
    def suggest_initial_solution(self, permut: ArrayLike) -> ArrayLike:
        """
        Generate an initial feasible solution based on the given permutation.

        Parameters
        ----------
        permut : ArrayLike
            Array representing the order of variables.

        Returns
        -------
        ArrayLike
            Initial feasible solution.
        """
        pass

    @abstractmethod
    def check_constraints(self, x: ArrayLike) -> bool:
        """
        Verify whether a given solution satisfies the constraints.

        Parameters
        ----------
        x : ArrayLike
            Solution array to be validated.

        Returns
        -------
        bool
            True if the solution satisfies the constraints, otherwise False.
        """
        pass

    @abstractmethod
    def suggest_solution(self, permut: ArrayLike, suggest_callback: Callable) -> ArrayLike:
        """
        Suggest a feasible solution using a callback function.

        Parameters
        ----------
        permut : ArrayLike
            Array representing the order of variables.
        suggest_callback : Callable
            Function that determines values of variables.

        Returns
        -------
        ArrayLike
            Suggested feasible solution.
        """
        pass

    @abstractmethod
    def get_ub(self, var_num: int) -> int:
        """
        Get the upper bound for a given variable.

        Parameters
        ----------
        var_num : int
            Index of the variable.

        Returns
        -------
        int
            Upper bound of the variable.
        """
        pass


class CapacityConstraints(Constraints):
    """
    Defines capacity constraints for optimization problems, ensuring that
    variable values do not exceed their predefined upper bounds.
    """

    def __init__(self, facade: BlocksNetFacade, num_params: int):
        """
        Initialize capacity constraints with upper bounds for each variable.

        Parameters
        ----------
        facade : BlocksNetFacade
            Interface providing constraint data.
        num_params : int
            Number of variables in the optimization problem.
        """
        self._upper_bounds = np.array([facade.get_upper_bound_var(var_num) for var_num in range(num_params)])
        if any(weight <= 0 for weight in self._upper_bounds):
            raise ValueError("All upper bounds of variables must be positive numbers.")

    def suggest_initial_solution(self, permut: ArrayLike) -> ArrayLike:
        """
        Generate an initial solution while respecting upper bounds.

        Parameters
        ----------
        permut : ArrayLike
            Array representing the order of variables.

        Returns
        -------
        ArrayLike
            Initial feasible solution.
        """
        x = np.zeros(permut.shape[0], dtype=int)
        for i in range(permut.shape[0]):
            chosen_val = min(1, self.get_ub(permut[i]))
            x[permut[i]] = chosen_val
        return x

    def suggest_solution(self, permut: ArrayLike, suggest_callback: Callable) -> ArrayLike:
        """
        Suggest a feasible solution using a callback function.

        Parameters
        ----------
        permut : ArrayLike
            Array representing the order of variables.
        suggest_callback : Callable
            Function that determines values of variables.

        Returns
        -------
        ArrayLike
            Suggested feasible solution.
        """
        x = np.zeros(permut.shape[0], dtype=int)
        for i in range(permut.shape[0]):
            lb = 0
            ub = self.get_ub(permut[i])
            val = suggest_callback(permut[i], lb, ub)
            x[permut[i]] = val
        return x

    def get_ub(self, var_num: int) -> int:
        """
        Get the upper bound for a specific variable.

        Parameters
        ----------
        var_num : int
            Index of the variable.

        Returns
        -------
        int
            Upper bound of the variable.
        """
        return self._upper_bounds[var_num]

    def check_constraints(self, x: ArrayLike) -> bool:
        """
        Validate if a given solution satisfies the capacity constraints.

        Parameters
        ----------
        x : ArrayLike
            Solution array to be validated.

        Returns
        -------
        bool
            True if constraints are satisfied, otherwise False.
        """
        return np.all(x <= self._upper_bounds)


class WeightedConstraints(Constraints):
    """
    Implements constraints based on weighted limits for grouped variables.
    """

    def __init__(self, facade: BlocksNetFacade, num_params: int):
        """
        Initializes WeightedConstraints with variable groups, group limits, and variable weights.

        Parameters
        ----------
        facade : BlocksNetFacade
            Interface providing constraint data.
        num_params : int
            Number of variables.
        """
        self._vars_group = np.array([facade.get_group_num(var_num) for var_num in range(num_params)])
        self._groups_nums = set(self._vars_group)
        self._groups_limit: Dict[int, float] = {group_num: facade.get_limit(group_num) for group_num in self._groups_nums}
        self._weights = np.array([facade.get_var_weight(var_num) for var_num in range(num_params)])
        if any(weight <= 0 for weight in self._weights):
            raise ValueError("All weights must be positive numbers.")

    def suggest_initial_solution(self, permut: ArrayLike) -> ArrayLike:
        """
        Suggests an initial solution while ensuring group weight limits are respected.

        Parameters
        ----------
        permut : ArrayLike
            Array representing the order of variables.

        Returns
        -------
        ArrayLike
            Initial feasible solution.
        """
        group_sums = {group_num: 0 for group_num in self._groups_nums}
        x = np.zeros(permut.shape[0], dtype=int)
        for var_num in permut:
            var_group = self._vars_group[var_num]
            hi = np.floor((self._groups_limit[var_group] - group_sums[var_group]) / self._weights[var_num])
            chosen_val = min(1, hi)
            group_sums[var_group] += self._weights[var_num] * chosen_val
            x[var_num] = chosen_val
        return x

    def suggest_solution(self, permut: ArrayLike, suggest_callback: Callable) -> ArrayLike:
        """
        Suggests a solution using a callback function while respecting group constraints.

        Parameters
        ----------
        permut : ArrayLike
            Array representing the order of variables.
        suggest_callback : Callable
            Function to determine values of variables.

        Returns
        -------
        ArrayLike
            Suggested feasible solution.
        """
        group_sums = {group_num: 0 for group_num in self._groups_nums}
        x = np.zeros(permut.shape[0], dtype=int)
        for var_num in permut:
            var_group = self._vars_group[var_num]
            ub = np.floor((self._groups_limit[var_group] - group_sums[var_group]) / self._weights[var_num])
            lb = 0
            ub = max(0, ub)
            val = suggest_callback(var_num, lb, ub)
            group_sums[var_group] += self._weights[var_num] * val
            x[var_num] = val
        return x

    def get_ub(self, var_num: int) -> int:
        """
        Returns the upper bound for a given variable based on its group constraints.

        Parameters
        ----------
        var_num : int
            Index of the variable.

        Returns
        -------
        int
            Upper bound for the variable.
        """
        var_group = self._vars_group[var_num]
        return int(np.floor(self._groups_limit[var_group] / self._weights[var_num]))

    def check_constraints(self, x: ArrayLike) -> bool:
        """
        Checks whether the given solution satisfies the group constraints.

        Parameters
        ----------
        x : ArrayLike
            Solution array to be validated.

        Returns
        -------
        bool
            True if constraints are satisfied, otherwise False.
        """
        group_sums = {group: np.sum(x[self._vars_group == group] * self._weights[self._vars_group == group]) for group in self._groups_nums}
        return all(group_sums[group] <= self._groups_limit[group] for group in self._groups_nums)

