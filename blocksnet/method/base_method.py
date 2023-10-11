from abc import ABC
from pydantic import BaseModel, InstanceOf
from ..models import City


class BaseMethod(ABC, BaseModel):
    """BaseMethod class required for methods implementation"""

    city_model: InstanceOf[City]

    def calculate(self, *args, **kwargs) -> any:
        """Main calculation method that should be overrided in child class"""
        raise NotImplementedError("Calculation method is not implemented")

    def plot(self, *args, **kwargs) -> any:
        """Plot method that can be overrided in child class"""
        raise NotImplementedError("Plot method is not implemented")
