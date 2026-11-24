import logging
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional

import numpy as np
from numpy.typing import ArrayLike

from blocksnet.optimization.services.acl import Facade


class VariableChooser(ABC):
    """
    Abstract base class for selecting optimization variables in urban planning optimization.

    This class defines the interface for variable selection strategies that determine which
    variables (representing potential service placements) should be prioritized during
    the optimization process.

    Attributes
    ----------
    _facade : Facade
        The facade instance providing access to city data and optimization methods.
    """

    def __init__(self, facade: Facade):
        """
        Initialize the VariableChooser with a facade instance.

        Parameters
        ----------
        facade : Facade
            The facade providing access to city data and optimization methods.
        """
        self._facade = facade

    @abstractmethod
    def _choose(self, x: ArrayLike, trials_data_callback: Callable) -> tuple[ArrayLike, ArrayLike]:
        """
        Abstract method for implementing variable selection logic.

        Parameters
        ----------
        x : ArrayLike
            Current solution array or permutation of variables.
        trials_data_callback : Callable
            Function returning trial data (previous solutions and their provisions).

        Returns
        -------
        ArrayLike
            Array of selected variable indices.
        """
        pass

    def __call__(self, x: ArrayLike, trials_data_callback: Callable) -> ArrayLike:
        """
        Callable interface for variable selection.

        Parameters
        ----------
        x : ArrayLike
            Current solution array or permutation of variables.
        trials_data_callback : Callable
            Function returning trial data (previous solutions and their provisions).

        Returns
        -------
        ArrayLike
            Array of selected variable indices.
        """
        return self._choose(x, trials_data_callback)


class SimpleChooser(VariableChooser):
    """Chooser that optimizes every variable and fixes none."""

    def __init__(self, facade: Facade = None):
        super().__init__(facade)

    def _choose(self, permut: ArrayLike, trials_data_callback: Callable) -> tuple[ArrayLike, ArrayLike]:
        """
        Select variables based on service type weights.

        Parameters
        ----------
        permut : ArrayLike
            Permutation of variable indices to consider.
        trials_data_callback : Callable
            Unused in this implementation (maintained for interface consistency).

        Returns
        -------
        ArrayLike
            Array of selected variable indices, prioritizing higher-weighted services.
        """

        return permut, np.array([])


class WeightChooser(VariableChooser):
    """
    Variable selection strategy based on predefined service type weights.

    This chooser prioritizes variables representing services with higher weights,
    selecting a limited number of top services per urban block.

    Attributes
    ----------
    _num_top : int
        Maximum number of top services to select per block.
    _weights : Dict[str, float]
        Dictionary mapping service types to their selection weights.
    """

    def __init__(self, facade: Facade, weights: Dict[str, float], num_top: int = 5):
        """
        Initialize the WeightChooser with service weights and selection parameters.

        Parameters
        ----------
        facade : Facade
            The facade providing access to city data.
        weights : Dict[str, float]
            Dictionary mapping service types to their selection weights.
        num_top : int, optional
            Maximum number of top services to select per block (default: 5).
        """
        super().__init__(facade)
        self._num_top = num_top
        self._weights = weights

    def _choose(self, permut: ArrayLike, trials_data_callback: Callable) -> tuple[ArrayLike, ArrayLike]:
        """
        Select variables based on service type weights.

        Parameters
        ----------
        permut : ArrayLike
            Permutation of variable indices to consider.
        trials_data_callback : Callable
            Unused in this implementation (maintained for interface consistency).

        Returns
        -------
        ArrayLike
            Array of selected variable indices, prioritizing higher-weighted services.
        """
        services = set()
        for var_num in permut:
            var_service = self._facade.get_service_name(var_num)
            services.add(var_service)
        # Group services by block and sort by weight
        services_list = list(services)
        services_list.sort(key=lambda service_type: -self._weights[service_type])

        n = len(services)
        opt_threshold = min(int(np.ceil(n / 3)), self._num_top)

        return np.array(
            [var_num for var_num in permut if self._facade.get_service_name(var_num) in services_list[:opt_threshold]]
        ), np.array(
            [var_num for var_num in permut if self._facade.get_service_name(var_num) in services_list[opt_threshold:]]
        )


class GradientChooser(VariableChooser):
    """
    Variable selection strategy based on gradient information from previous trials.

    This chooser prioritizes variables that showed the most improvement in provision
    per unit of resource invested during previous optimization trials.

    Attributes
    ----------
    _num_top : int
        Maximum number of top services to select per block.
    _num_params : int
        Total number of variables in the optimization problem.
    """

    def __init__(
        self, facade: Facade, num_params: int, num_top: int = 5, weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the GradientChooser with selection parameters.

        Parameters
        ----------
        facade : Facade
            The facade providing access to city data.
        num_params : int
            Total number of variables in the optimization problem.
        num_top : int, optional
            Maximum number of top services to select per block (default: 5).
        weights : Optional[Dict[str, float]]
            Objective weight per service type. The objective is a weighted sum
            of provisions, so a service's contribution is its provision gain
            times its weight. Without them every type counts the same.
        """
        super().__init__(facade)
        self._num_top = num_top
        self._num_params = num_params
        self._weights = weights or {}
        self._explore_cursor = 0

    def _take_untried(self, untried: List[str], count: int) -> List[str]:
        """
        Pick service types that have not been moved yet, a few per trial.

        A type only earns a gradient once it has been changed, and it can only
        be changed once it is chosen, so without a reserved share of the slots
        the set of types explored is decided by the first trials and never
        grows: over runs of 39 and 96 trials the solution held the same six
        types, and schools and kindergartens were never among them.

        Parameters
        ----------
        untried : List[str]
            Service types with no movement to measure a gradient from.
        count : int
            How many of them to return.

        Returns
        -------
        List[str]
            The next ``count`` of them, continuing where the last call stopped.
        """
        if not untried or count <= 0:
            return []
        picked = [untried[(self._explore_cursor + i) % len(untried)] for i in range(min(count, len(untried)))]
        self._explore_cursor = (self._explore_cursor + len(picked)) % len(untried)
        return picked

    def _choose(self, permut: ArrayLike, trials_data_callback: Callable) -> tuple[ArrayLike, ArrayLike]:
        """
        Select variables based on gradient information from previous trials.

        Parameters
        ----------
        permut : ArrayLike
            Permutation of variable indices to consider.
        trials_data_callback : Callable
            Function returning trial data (previous solutions and their provisions).

        Returns
        -------
        ArrayLike
            Array of selected variable indices, prioritizing services with highest
            provision improvement per resource invested.
        """
        second_last_trial, last_trial = trials_data_callback()  # obj_val, x, provisions
        var_area = [sum(self._facade.get_var_weights(var_num)[0:2]) for var_num in range(self._num_params)]

        # Area each service type gained or gave up between the last two trials.
        services = []
        spent = {}
        for var_num in permut:
            var_service = self._facade.get_service_name(var_num)
            if var_service not in spent:
                spent[var_service] = 0.0
                services.append(var_service)
            spent[var_service] += (last_trial[1][var_num] - second_last_trial[1][var_num]) * var_area[var_num]

        last_provisions = last_trial[2] or {}
        previous_provisions = second_last_trial[2] or {}

        def gradient(service_type: str) -> float:
            """What one square metre spent on this service type bought.

            The provision of this type, not the objective as a whole: the
            objective's change is one number shared by every type, so dividing
            it by each type's area ranked them by who spent least, and a school
            on a 33000 m2 plot could never outrank a 10 m2 bus stop.
            """
            gain = last_provisions.get(service_type, 0.0) - previous_provisions.get(service_type, 0.0)
            return gain * self._weights.get(service_type, 1.0) / abs(spent[service_type])

        tried = [service_type for service_type in services if spent[service_type] != 0]
        untried = [service_type for service_type in services if spent[service_type] == 0]
        tried.sort(key=lambda service_type: -gradient(service_type))

        n = len(services)
        opt_threshold = min(int(np.ceil(n / 2)), self._num_top)
        # Keep part of the budget for types never moved yet, so the set of
        # types under optimization can still grow after the first trials.
        explore = min(len(untried), opt_threshold // 2) if untried else 0
        chosen = set(tried[: opt_threshold - explore]) | set(self._take_untried(untried, explore))

        # Everything else is held at its current value. It used to be that the
        # worst-ranked types fell out of both lists, and the optimizer then
        # pinned them to zero on every trial.
        return np.array([var_num for var_num in permut if self._facade.get_service_name(var_num) in chosen]), np.array(
            [var_num for var_num in permut if self._facade.get_service_name(var_num) not in chosen]
        )
