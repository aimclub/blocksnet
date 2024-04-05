import math

from pydantic import BaseModel, Field, field_validator
from ..utils import SQUARE_METERS_IN_HECTARE


class ServiceBrick(BaseModel):
    """Typical service type brick or chunk, that is used in development"""

    capacity: int
    """How many people in need can be supported by a service"""
    area: float
    """Area in hectares"""
    is_integrated: bool
    """Is integrated within the building"""
    parking_area: float

    @property
    def sq_m_area(self):
        """Self area in square meters"""
        return self.area * SQUARE_METERS_IN_HECTARE


class ServiceType(BaseModel):
    """Represents service type entity, such as schools and its parameters overall"""

    code: str
    name: str
    accessibility: int = Field(gt=0)
    demand: int = Field(gt=0)
    bricks: list[ServiceBrick] = []

    @field_validator("bricks", mode="before")
    def validate_bricks(value):
        bricks = [sb if isinstance(sb, ServiceBrick) else ServiceBrick(**sb) for sb in value]
        return bricks

    def calculate_in_need(self, population: int) -> int:
        """Calculate how many people in the given population are in need by this service type"""
        return math.ceil(population / 1000 * self.demand)

    def __hash__(self):
        """Make service type hashable to use it as a key"""
        return hash(self.name)

    def __str__(self):
        accessibility = f"{self.accessibility} min"
        demand = f"{self.demand}/1000 population"
        return f"{self.code.ljust(10)} {self.name.ljust(20)} {accessibility.ljust(10)} {demand.ljust(20)}"
