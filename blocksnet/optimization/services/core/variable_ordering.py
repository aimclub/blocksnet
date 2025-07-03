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
        """
        Abstract method to define the ordering strategy.

        Parameters
        ----------
        x : ArrayLike
            Input array to be ordered.

        Returns
        -------
        ArrayLike
            Ordered array according to the specific strategy.
        """
        pass

    def __call__(self, x: ArrayLike) -> ArrayLike:
        """
        Allow the instance to be called as a function.

        Parameters
        ----------
        x : ArrayLike
            Input array to be ordered.

        Returns
        -------
        ArrayLike
            Result of applying the ordering strategy to the input.
        """
        return self._order(x)


class NeutralOrder(VariablesOrder):
    """
    Class that applies no ordering to the input data.

    This class represents a neutral ordering strategy where the input data is returned
    without any modification.
    """

    def _order(self, x: ArrayLike) -> ArrayLike:
        """
        Return the input data unchanged.

        Parameters
        ----------
        x : ArrayLike
            Input array to be processed.

        Returns
        -------
        ArrayLike
            The same input array, unmodified.
        """
        return x


class RandomOrder(VariablesOrder):
    """
    Class that orders the input data randomly.

    This class applies a random shuffle to the input data, altering its order in place.
    The shuffle is performed using numpy.random.shuffle.
    """

    def __init__(self):
        """Initialize the random order strategy."""
        super().__init__()

    def _order(self, x: ArrayLike) -> ArrayLike:
        """
        Randomly shuffle the input array in-place.

        Parameters
        ----------
        x : ArrayLike
            Input array to be shuffled.

        Returns
        -------
        ArrayLike
            The shuffled array (same object as input, modified in-place).
        """
        shuffle(x)
        return x


class AscendingOrder(VariablesOrder):
    """
    Class that orders the input data in ascending order.

    This class sorts the input data in increasing order using Python's built-in sorted().
    """

    def _order(self, x: ArrayLike) -> ArrayLike:
        """
        Sort the input array in ascending order.

        Parameters
        ----------
        x : ArrayLike
            Input array to be sorted.

        Returns
        -------
        ArrayLike
            New array containing the sorted elements in ascending order.
        """
        return sorted(x)


class DescendingOrder(VariablesOrder):
    """
    Class that orders the input data in descending order.

    This class sorts the input data in decreasing order using Python's built-in sorted()
    with reverse=True.
    """

    def _order(self, x: ArrayLike) -> ArrayLike:
        """
        Sort the input array in descending order.

        Parameters
        ----------
        x : ArrayLike
            Input array to be sorted.

        Returns
        -------
        ArrayLike
            New array containing the sorted elements in descending order.
        """
        return sorted(x, reverse=True)


class IndexBasedOrder(VariablesOrder):
    """
    Class that orders the input data based on a provided list of indices.

    This class takes a predefined list of indices and reorders the input data
    accordingly, selecting elements from the input based on the specified indices.
    """

    def __init__(self, indices: ArrayLike):
        """
        Initialize the index-based order strategy.

        Parameters
        ----------
        indices : ArrayLike
            List of indices specifying the desired order of the input data.
        """
        super().__init__()
        self.indices = indices

    def _order(self, x: ArrayLike) -> ArrayLike:
        """
        Order the input array based on the provided indices.

        Parameters
        ----------
        x : ArrayLike
            Input array to be reordered.

        Returns
        -------
        ArrayLike
            New array containing elements from the input array in the order specified
            by the indices.

        Raises
        ------
        ValueError
            If the length of indices does not match the length of input array.
        """
        if len(self.indices) != len(x):
            raise ValueError("Length of indices must match the length of x.")
        return [x[i] for i in self.indices]
