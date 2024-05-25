from abc import ABC

from pydantic import BaseModel, InstanceOf

from ..models.region import Region


class BaseMethod(ABC, BaseModel):
    """BaseMethod class required for methods implementation"""

    region: InstanceOf[Region]
