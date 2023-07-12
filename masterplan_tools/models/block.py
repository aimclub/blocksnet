from pydantic import BaseModel, Field
from .geojson import Geometry


class Block(BaseModel):
    """
    A class representing block entity
    """

    id: int = Field(ge=0)
    geojson: Geometry
