import copy
import logging
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional

import numpy as np
from numpy.typing import ArrayLike

from blocksnet.enums import LandUse
from blocksnet.optimization.services.acl import Facade
from blocksnet.optimization.services.common.variable import Variable


class Constraints(ABC):
    """
    Abstract base class defining a set of constraints for optimization problems.
    """

    def __init__(self, facade: Facade, num_params: int):
        """
        Initialize the Constraints base class.

        Parameters
        ----------
        facade : Facade
            Interface providing access to optimization problem data.
        num_params : int
            Number of parameters/variables in the optimization problem.
        """
        self._facade = facade
        self._num_params = num_params
        self._prev_solution: ArrayLike = None

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

    def update_solution(self, x: ArrayLike):
        """
        Update the stored previous solution.

        Parameters
        ----------
        x : ArrayLike
            Current solution to store as previous solution.
        """
        if self._prev_solution is None:
            self._prev_solution = copy.deepcopy(x)
        else:
            self._prev_solution = None

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
    def correct_X(self, x: ArrayLike) -> ArrayLike:
        """
        Correct/adjust a solution to ensure it meets constraints.

        Parameters
        ----------
        x : ArrayLike
            Solution array to be corrected.

        Returns
        -------
        ArrayLike
            Corrected solution array.
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
        super().__init__(facade, num_params)
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

    def correct_X(self, x: ArrayLike) -> ArrayLike:
        """
        Convert solution array to list of Variable objects.

        Parameters
        ----------
        x : ArrayLike
            Solution array to convert.

        Returns
        -------
        ArrayLike
            List of Variable objects representing the solution.
        """
        return self._facade.get_X(x)


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
        super().__init__(facade, num_params)
        self._decrease_ub_coef = decrease_ub_coef
        self._vars_block = np.array([facade.get_var_block_id(var_num) for var_num in range(num_params)])
        self._vars_services = np.array([facade.get_service_name(var_num) for var_num in range(num_params)])
        self._block_ids = set(self._vars_block)
        self._weights = np.array([facade.get_var_weights(var_num) for var_num in range(num_params)])
        self._blocks_services = {block_id: facade.get_block_services(block_id) for block_id in self._block_ids}
        self._priority = (
            sorted([service_type for service_type in priority.keys()], key=lambda x: priority[x], reverse=True)
            if priority
            else []
        )

    def _initialize_blocks_limits(self, permut: ArrayLike) -> tuple[Dict[int, List], Dict[int, Dict[str, float]]]:
        """
        Initialize block limits for area and capacity constraints.

        Parameters
        ----------
        permut : ArrayLike
            Array representing the order of variables.

        Returns
        -------
        tuple[Dict[int, List], Dict[int, Dict[str, float]]]
            Tuple containing:
            - Dictionary of area limits (site area and build floor area) per block
            - Dictionary of capacity limits per service per block
        """
        limits_area = {block_id: [0, 0] for block_id in self._block_ids}
        limits_capacity = {
            block_id: {service: 0 for service in self._blocks_services[block_id]} for block_id in self._block_ids
        }
        for var_num in permut:
            var_block = self._vars_block[var_num]
            var_service = self._vars_services[var_num]
            if np.count_nonzero(limits_area[var_block]) == 0:
                limits_area[var_block] = self._facade.get_limits_var(var_num)[:2]

            if limits_capacity[var_block][var_service] == 0:
                limits_capacity[var_block][var_service] = self._facade.get_limits_var(var_num)[2]
        return limits_area, limits_capacity

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
        limits_area, limits_capacity = self._initialize_blocks_limits(permut)

        x = np.zeros(self._num_params, dtype=int)

        weights = self._weights
        vars_block = self._vars_block
        vars_services = self._vars_services
        facade = self._facade

        for var_num in permut:
            lb = facade.get_lower_bound_var_val(var_num)
            if lb == 0:
                continue

            b = vars_block[var_num]
            s = vars_services[var_num]
            w0, w1, w2 = weights[var_num]

            fits_sa = (w0 == 0) or (w0 * lb <= limits_area[b][0])
            fits_bfa = (w1 == 0) or (w1 * lb <= limits_area[b][1])
            fits_cap = (w2 == 0) or (w2 * lb <= limits_capacity[b][s])

            if fits_sa and fits_bfa and fits_cap:
                x[var_num] = lb
                limits_area[b][0] -= w0 * lb
                limits_area[b][1] -= w1 * lb
                limits_capacity[b][s] -= w2 * lb

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
        limits_area, limits_capacity = self._initialize_blocks_limits(permut)
        x = np.zeros(self._num_params, dtype=int)
        logging.info(f"Suggesting new solution. Limits: area (sa/bfa) = {limits_area}, capacity = {limits_capacity}")
        logging.info(f"Suggesting new solution. Previous solution: {self._prev_solution}")
        for var_num in permut:
            var_block = self._vars_block[var_num]
            var_service = self._vars_services[var_num]
            prev_val = 0 if self._prev_solution is None else self._prev_solution[var_num]
            limits_area[var_block][0] -= self._weights[var_num][0] * prev_val
            if (
                self._facade.get_var_land_use(var_num) == LandUse.RESIDENTIAL
                and np.count_nonzero(self._prev_solution) > 0
            ):
                limits_area[var_block][1] = 0
            else:
                limits_area[var_block][1] -= self._weights[var_num][1] * prev_val
            limits_capacity[var_block][var_service] -= self._weights[var_num][2] * prev_val

        for var_num in permut:
            var_block = self._vars_block[var_num]
            var_service = self._vars_services[var_num]
            prev_val = 0 if self._prev_solution is None else self._prev_solution[var_num]
            lb = prev_val
            bound_range = self.get_ub(var_num) - lb

            if self._weights[var_num][0] > 0:
                bound_range = min(bound_range, np.floor(limits_area[var_block][0] / self._weights[var_num][0]))
            if self._weights[var_num][1] > 0:
                bound_range = min(bound_range, np.floor(limits_area[var_block][1] / self._weights[var_num][1]))
            if self._weights[var_num][2] > 0:
                bound_range = min(
                    bound_range, np.floor(limits_capacity[var_block][var_service] / self._weights[var_num][2])
                )

            bound_range = max(bound_range, 0)  # On initial solution bound range can be < 0, but solution is still valid

            val = suggest_callback(var_num, lb, lb + bound_range)
            x[var_num] = val

            addition = x[var_num] - lb
            limits_area[var_block][0] -= self._weights[var_num][0] * addition
            limits_area[var_block][1] -= self._weights[var_num][1] * addition
            limits_capacity[var_block][var_service] -= self._weights[var_num][2] * addition
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
        return self._facade.get_upper_bound_var(var_num)

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

    def correct_X(self, x: ArrayLike) -> List[Variable]:
        """
        Convert solution array to list of Variable objects.

        Parameters
        ----------
        x : ArrayLike
            Solution array to convert.

        Returns
        -------
        List[Variable]
            List of Variable objects representing the solution.
        """
        return self._facade.get_X(x)
