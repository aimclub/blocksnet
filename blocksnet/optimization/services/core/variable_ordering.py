from abc import ABC, abstractmethod

from numpy.random import shuffle
from numpy.typing import ArrayLike


class VariablesOrder(ABC):
    """
    Abstract base class for defining different strategies for ordering variables.

    This class provides an abstract method `_order` which must be implemented by subclasses
    to define the specific ordering strategy. The `__call__` method allows instances of
    this class to be used as functions, applying the ordering strategy to input data.
    """

    def __init__(self):
        pass

    @abstractmethod
    def _order(self, x: ArrayLike) -> ArrayLike:
        pass

    def __call__(self, x: ArrayLike) -> ArrayLike:
        return self._order(x)


class NeutralOrder(VariablesOrder):
    """
    Class that applies no ordering to the input data.

    This class represents a neutral ordering strategy where the input data is returned
    without any modification.
    """

    def _order(self, x: ArrayLike) -> ArrayLike:
        """
        Returns the input data as is (no change).

        Parameters
        ----------
        x : ArrayLike
            Input data.

        Returns
        -------
        ArrayLike
            The same input data, unmodified.
        """
        return x


class RandomOrder(VariablesOrder):
    """
    Class that orders the input data randomly.

    This class applies a random shuffle to the input data, altering its order in place.
    """

    def __init__(self):
        super().__init__()

    def _order(self, x: ArrayLike) -> ArrayLike:
        """
        Randomly shuffles the input data.

        Parameters
        ----------
        x : ArrayLike
            Input data to be shuffled.

        Returns
        -------
        ArrayLike
            The shuffled input data.
        """
        shuffle(x)
        return x


class AscendingOrder(VariablesOrder):
    """
    Class that orders the input data in ascending order.

    This class sorts the input data in increasing order.
    """

    def _order(self, x: ArrayLike) -> ArrayLike:
        """
        Sorts the input data in ascending order.

        Parameters
        ----------
        x : ArrayLike
            Input data to be sorted.

        Returns
        -------
        ArrayLike
            The sorted data in ascending order.
        """
        return sorted(x)


class DescendingOrder(VariablesOrder):
    """
    Class that orders the input data in descending order.

    This class sorts the input data in decreasing order.
    """

    def _order(self, x: ArrayLike) -> ArrayLike:
        """
        Sorts the input data in descending order.

        Parameters
        ----------
        x : ArrayLike
            Input data to be sorted.

        Returns
        -------
        ArrayLike
            The sorted data in descending order.
        """
        return sorted(x, reverse=True)


class IndexBasedOrder(VariablesOrder):
    """
    Class that orders the input data based on a provided list of indices.

    This class takes a predefined list of indices and reorders the input data
    accordingly, selecting elements from the input based on the specified indices.

    Attributes
    ----------
    indices : ArrayLike
        List of indices specifying the desired order of the input data.
    """

    def __init__(self, indices: ArrayLike):
        super().__init__()
        self.indices = indices

    def _order(self, x: ArrayLike) -> ArrayLike:
        """
        Orders the input data based on the provided indices.

        Parameters
        ----------
        x : ArrayLike
            Input data to be reordered.

        Returns
        -------
        ArrayLike
            The reordered data based on the specified indices.

        Raises
        ------
        ValueError
            If the length of indices does not match the length of input data.
        """
        if len(self.indices) != len(x):
            raise ValueError("Length of indices must match the length of x.")
        return [x[i] for i in self.indices]
