from abc import ABC, abstractmethod
from typing import Callable, Dict

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
    def __init__(self, facade: Facade):
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
        services_list.sort(lambda x: -self._weights[x])

        n = len(permut)
        opt_threshold = min(int(np.ceil(n / 3)), self._num_top)
        null_threshold = min(int(np.ceil(n / 4)), self._num_top)

        return np.array(
            [var_num for var_num in permut if self._facade.get_service_name(var_num) in services_list[:opt_threshold]]
        ), np.array(
            [
                var_num
                for var_num in permut
                if self._facade.get_service_name(var_num) in services_list[opt_threshold:-null_threshold]
            ]
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

    def __init__(self, facade: Facade, num_params: int, num_top: int = 5):
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
        """
        super().__init__(facade)
        self._num_top = num_top
        self._num_params = num_params

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
        # Get trial data and filter variables with remaining capacity
        second_last_trial, last_trial = trials_data_callback()
        new_permut = np.array([var_num for var_num in permut if self._facade.get_upper_bound_var(var_num) > 0])
        weights = [self._facade.get_var_weights(var_num)[0] for var_num in range(self._num_params)]

        # Initialize data structures for tracking service improvements
        services = []
        service_increment = {}

        # Calculate total resource investment and provision improvement per service
        for var_num in new_permut:
            var_service = self._facade.get_service_name(var_num)
            diff = last_trial[1][var_num] - second_last_trial[1][var_num]

            if var_service not in service_increment:
                service_increment[var_service] = 0.0
                services.append(var_service)

            service_increment[var_service] += diff * weights[var_num]

        # Define sorting key based on provision improvement per resource invested
        def gradient_sortkey(x):
            if service_increment[x] == 0:
                return 0

            provision_diff = last_trial[0]
            if second_last_trial[0] is not None:
                provision_diff -= second_last_trial[0]

            return -provision_diff / service_increment[x]

        # Sort services by their efficiency (provision improvement per resource)
        services.sort(key=gradient_sortkey)

        n = len(permut)
        opt_threshold = min(int(np.ceil(n / 3)), self._num_top)
        null_threshold = min(int(np.ceil(n / 4)), self._num_top)

        # Select top N services per block
        return np.array(
            [var_num for var_num in permut if self._facade.get_service_name(var_num) in services[:opt_threshold]]
        ), np.array(
            [
                var_num
                for var_num in permut
                if self._facade.get_service_name(var_num) in services[opt_threshold:-null_threshold]
            ]
        )
