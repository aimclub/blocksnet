from pydantic import BaseModel, Field
from .geojson import Geometry


class Block(BaseModel):
    """
    A class representing city block entity
    """

    id: int = Field(ge=0)
    geometry: Geometry
