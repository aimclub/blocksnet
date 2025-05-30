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
    def _choose(self, x: ArrayLike, trials_data_callback: Callable) -> ArrayLike:
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

    def _choose(self, permut: ArrayLike, trials_data_callback: Callable) -> ArrayLike:
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
        # Group services by block and sort by weight
        block_services = {
            self._facade.get_var_block_id(var_num): sorted(
                self._facade.get_block_services(self._facade.get_var_block_id(var_num)),
                key=lambda x: -self._weights[x]
            )
            for var_num in permut
        }

        # Select top N services per block
        return np.array([
            var_num
            for var_num in permut
            if self._facade.get_service_name(var_num)
               in block_services[self._facade.get_var_block_id(var_num)][:self._num_top]
        ])


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

    def _choose(self, permut: ArrayLike, trials_data_callback: Callable) -> ArrayLike:
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
        first_trial, last_trial = trials_data_callback()
        new_permut = np.array([
            var_num for var_num in permut 
            if self._facade.get_upper_bound_var(var_num) > 0
        ])
        weights = [self._facade.get_var_weight(var_num) for var_num in range(self._num_params)]

        # Initialize data structures for tracking service improvements
        block_ids = {self._facade.get_var_block_id(var_num) for var_num in range(self._num_params)}
        block_services = {block_id: [] for block_id in block_ids}
        service_increment = {block_id: dict() for block_id in block_ids}

        # Calculate total resource investment and provision improvement per service
        for var_num in new_permut:
            var_block = self._facade.get_var_block_id(var_num)
            var_service = self._facade.get_service_name(var_num)
            diff = last_trial[1][var_num] - first_trial[1][var_num]
            
            if var_service not in service_increment[var_block]:
                service_increment[var_block][var_service] = 0.0
                block_services[var_block].append(var_service)
            
            service_increment[var_block][var_service] += diff * weights[var_num]

        # Define sorting key based on provision improvement per resource invested
        def gradient_sortkey(x):
            if service_increment[block_id][x] == 0:
                return 0
            
            provision_diff = last_trial[0]
            if first_trial[0] is not None:
                provision_diff -= first_trial[0]
            
            return -provision_diff / service_increment[block_id][x]

        # Sort services by their efficiency (provision improvement per resource)
        for block_id in block_services:
            block_services[block_id].sort(key=gradient_sortkey)

        # Select top N most efficient services per block
        return np.array([
            var_num
            for var_num in new_permut
            if self._facade.get_service_name(var_num)
               in block_services[self._facade.get_var_block_id(var_num)][:self._num_top]
        ])