from abc import ABC

from pydantic import BaseModel, InstanceOf

from ..models.region import Region


class BaseMethod(ABC, BaseModel):
    """BaseMethod class required for methods implementation"""

    region: InstanceOf[Region]

    def calculate(self, *args, **kwargs) -> any:
        """Main calculation method that should be overrided in child class"""
        raise NotImplementedError("Calculation method is not implemented")

    def plot(self, *args, **kwargs) -> any:
        """Plot method that can be overrided in child class"""
        raise NotImplementedError("Plot method is not implemented")
