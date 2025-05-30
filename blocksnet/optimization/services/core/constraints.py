from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional

import numpy as np
from numpy.typing import ArrayLike

from blocksnet.optimization.services.acl import Facade


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
    def suggest_fixed(self, permut: ArrayLike, suggest_callback: Callable) -> ArrayLike:
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

    @abstractmethod
    def decrease_ubs(self):
        """
        Decrease the upper bounds of variables, typically used when no feasible solution is found.
        """
        pass


class CapacityConstraints(Constraints):
    """
    Defines capacity constraints for optimization problems, ensuring that
    variable values do not exceed their predefined upper bounds.
    """

    def __init__(self, facade: Facade, num_params: int):
        """
        Initialize capacity constraints with upper bounds for each variable.

        Parameters
        ----------
        facade : Facade
            Interface providing constraint data.
        num_params : int
            Number of variables in the optimization problem.
        """
        self._facade = facade
        self._num_params = num_params
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
        x = np.zeros(self._num_params, dtype=int)
        for var_num in permut:
            chosen_val = min(1, self.get_ub(var_num))
            x[var_num] = chosen_val
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
        x = np.zeros(self._num_params, dtype=int)
        for var_num in permut:
            lb = 0
            ub = self.get_ub(var_num)
            val = suggest_callback(var_num, lb, ub)
            x[var_num] = val
        return x

    def suggest_fixed(self, permut: ArrayLike, suggest_callback: Callable) -> ArrayLike:
        x = np.zeros(self._num_params, dtype=int)
        for var_num in permut:
            val = suggest_callback(var_num)
            x[var_num] = val
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

    def __init__(self, facade: Facade, num_params: int, priority: Optional[Dict] = None, decrease_ub_coef: int = 2):
        """
        Initializes WeightedConstraints with variable groups, group limits, and variable weights.

        Parameters
        ----------
        facade : Facade
            Interface providing constraint data.
        num_params : int
            Number of variables.
        priority : Optional[Dict]
            Optional dictionary specifying priority weights for services.
        decrease_ub_coef : int
            Coefficient by which to decrease upper bounds when no solution is found.
        """
        self._num_params = num_params
        self._decrease_ub_coef = decrease_ub_coef
        self._vars_block = np.array([facade.get_var_block_id(var_num) for var_num in range(num_params)])
        self._vars_services = np.array([facade.get_service_name(var_num) for var_num in range(num_params)])
        self._upper_bounds = np.array([facade.get_upper_bound_var(var_num) for var_num in range(num_params)])
        self._facade = facade
        self._block_ids = set(self._vars_block)
        self._vars_limit: Dict[int, float] = {
            var_num: facade.get_limits_var(var_num) for var_num in range(self._num_params)
        }
        self._weights = np.array([facade.get_var_weights(var_num) for var_num in range(num_params)])
        self._priority: Dict[Dict] = {block_id: {} for block_id in self._block_ids}
        for block_id in self._block_ids:
            if priority is None:
                # Uniform distribution by default
                self._priority[block_id] = {
                    service: 1 / len(facade.get_block_services(block_id))
                    for service in facade.get_block_services(block_id)
                }
                continue

            # Need to do scaling to [0, 1] for algo
            total_priority = np.sum([priority[service] for service in facade.get_block_services(block_id)])

            self._priority[block_id] = {
                service: priority[service] / total_priority for service in facade.get_block_services(block_id)
            }

    def suggest_initial_solution1(self, permut: ArrayLike) -> ArrayLike:
        """
        Suggests an initial solution while ensuring group weight limits are respected.
        This is an alternative implementation that distributes capacity according to service priorities.

        Parameters
        ----------
        permut : ArrayLike
            Array representing the order of variables.

        Returns
        -------
        ArrayLike
            Initial feasible solution.
        """
        block_sums = {block_id: np.zeros(self._weights.shape[1]) for block_id in self._block_ids}
        block_services_sums = {
            block_id: {service: np.zeros(self._weights.shape[1]) for service in self._priority[block_id].keys()}
            for block_id in self._block_ids
        }
        x = np.zeros(self._num_params, dtype=int)
        for var_num in permut:
            var_block = self._vars_block[var_num]
            var_service = self._vars_services[var_num]

            chosen_val = min(
                [
                    np.floor(
                        (
                            self._vars_limit[var_num][i] * self._priority[var_block][var_service]
                            - block_services_sums[var_block][var_service][i]
                        )
                        / weight
                    )
                    for i, weight in enumerate(self._weights[var_num])
                ]
            )
            chosen_val = max(chosen_val, 0)
            chosen_val = min(chosen_val, self.get_ub(var_num))

            block_sums[var_block] += self._weights[var_num] * chosen_val
            block_services_sums[var_block][var_service] += self._weights[var_num] * chosen_val
            x[var_num] = chosen_val
        return x

    def suggest_initial_solution(self, permut: ArrayLike) -> ArrayLike:
        """
        Generate an initial solution while respecting upper bounds and ensuring
        at most one variable per service is selected in each block.

        Parameters
        ----------
        permut : ArrayLike
            Array representing the order of variables.

        Returns
        -------
        ArrayLike
            Initial feasible solution.
        """
        block_sums = {block_id: np.zeros(self._weights.shape[1]) for block_id in self._block_ids}
        added_services = set()
        x = np.zeros(self._num_params, dtype=int)
        for var_num in permut:
            var_block = self._vars_block[var_num]
            var_service = self._vars_services[var_num]
            if var_service in added_services:
                chosen_val = 0
            else:
                chosen_val = min(
                    np.floor((self._vars_limit[var_num][i] - block_sums[var_block][0]) / weight)
                    for i, weight in enumerate(self._weights[var_num])
                )
                chosen_val = max(0, chosen_val)
                chosen_val = min(chosen_val, self.get_ub(var_num))
                chosen_val = min(1, chosen_val)
                if chosen_val > 0:
                    added_services.add(var_service)
            x[var_num] = chosen_val
            block_sums[var_block] += self._weights[var_num] * chosen_val
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
        x = np.zeros(self._num_params, dtype=int)
        for var_num in permut:
            lb = 0
            ub = self.get_ub(var_num)
            if ub < lb:
                val = 0
            else:
                val = suggest_callback(var_num, lb, ub)
            x[var_num] = val
        return x

    def suggest_fixed(self, permut: ArrayLike, suggest_callback: Callable) -> ArrayLike:
        x = np.zeros(self._num_params, dtype=int)
        for var_num in permut:
            val = suggest_callback(var_num)
            x[var_num] = val
        return x

    def decrease_ubs(self):
        """
        Decrease the upper bounds of variables by the defined coefficient,
        with a minimum bound to prevent values from becoming too small.
        """
        for i in range(self._num_params):
            self._upper_bounds[i] = int(
                max(
                    np.ceil(self._upper_bounds[i] / self._decrease_ub_coef),
                    np.ceil(self._facade.get_upper_bound_var(i) / 10),
                )
            )

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
        return self._facade.check_constraints(x)
