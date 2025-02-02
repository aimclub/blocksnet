from abc import ABC, abstractmethod
from typing import Callable, Dict

import numpy as np
from numpy.typing import ArrayLike

from blocksnet.optimizer.src.acl import BlocksnetFacade


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

    def __init__(self, facade: BlocksnetFacade, num_params: int):
        """
        Initialize capacity constraints with upper bounds for each variable.

        Parameters
        ----------
        facade : BlocksnetFacade
            Interface providing constraint data.
        num_params : int
            Number of variables in the optimization problem.
        """
        pass
    
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
        pass

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
        pass

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
        pass


class WeightedConstraints(Constraints):
    """
    Implements constraints based on weighted limits for grouped variables.
    """

    def __init__(self, facade: BlocksnetFacade, num_params: int):
        """
        Initializes WeightedConstraints with variable groups, group limits, and variable weights.

        Parameters
        ----------
        facade : BlocksnetFacade
            Interface providing constraint data.
        num_params : int
            Number of variables.
        """
        pass

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
        pass

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
        pass

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
        pass

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
        pass
