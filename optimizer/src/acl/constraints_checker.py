from typing import Dict, List
from abc import ABC
from numpy.typing import ArrayLike

from blocksnet.method.annealing_optimizer import Indicator, Variable


class Constraints(ABC):
    """
    Abstract class defining methods for checking constraints of solution
    for a set of variables.
    """
    def __init__(self):
        super().__init__() #TODO: complete base class
    
    @abstractmethod
    def check_constraints(self, X: List[Variable], indicators: Dict[int, Indicator]) -> bool:
        """
        Check if the given solution satisfies all constraints.

        Parameters
        ----------
        X : List[Variable]
            List of variables representing the current solution.
        indicators : Dict[int, Indicator]
            Dictionary mapping block IDs to their corresponding indicators.

        Returns
        -------
        bool
            True if all constraints are satisfied, False otherwise.
        """
        pass


class AreaChecker(Constraints):
    """
    Implementation of Constraints that performs constraint checking and distance calculations
    for a given solution of variables and corresponding indicators.
    """
    pass  # TODO: Implement area-specific constraint logic
