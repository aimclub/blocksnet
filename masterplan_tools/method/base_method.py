from abc import ABC
from pydantic import BaseModel, InstanceOf
from ..models import City


class BaseMethod(ABC, BaseModel):
    """BaseMethod class required for methods implementation"""

    city_model: InstanceOf[City]
