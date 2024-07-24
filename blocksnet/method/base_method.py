from abc import ABC

from pydantic import BaseModel, InstanceOf

from ..models import City


class BaseMethod(ABC, BaseModel):
    """
    BaseMethod class required for methods implementation.

    Attributes
    ----------
    city_model : InstanceOf[City]
        Instance of the City model used in the method.

    Methods
    -------
    calculate(*args, **kwargs) -> any
        Main calculation method that should be overridden in child class.

    plot(*args, **kwargs) -> any
        Plot method that can be overridden in child class.
    """

    city_model: InstanceOf[City]

    def calculate(self, *args, **kwargs) -> any:
        """
        Main calculation method that should be overridden in child class.

        Parameters
        ----------
        *args : tuple
            Variable length argument list.

        **kwargs : dict
            Arbitrary keyword arguments.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in the child class.
        """
        raise NotImplementedError("Calculation method is not implemented")

    def plot(self, *args, **kwargs) -> any:
        """
        Plot method that can be overridden in child class.

        Parameters
        ----------
        *args : tuple
            Variable length argument list.

        **kwargs : dict
            Arbitrary keyword arguments.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in the child class.
        """
        raise NotImplementedError("Plot method is not implemented")
