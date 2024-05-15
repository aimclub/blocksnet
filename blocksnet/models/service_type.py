import math

from pydantic import BaseModel, Field, field_validator
from .land_use import LandUse


class ServiceBrick(BaseModel):
    """Typical service type brick or chunk, that is used in development"""

    capacity: int
    """How many people in need can be supported by a service"""
    area: float
    """Area in square meters"""
    is_integrated: bool
    """Is integrated within the building"""
    parking_area: float
    """Required parking area in square meters"""


class ServiceType(BaseModel):
    """Represents service type entity, such as schools and its parameters overall"""

    code: str
    name: str
    accessibility: int = Field(ge=0)
    demand: int = Field(ge=0)
    bricks: list[ServiceBrick] = []
    land_use: list[LandUse] = []

    def get_bricks(self, is_integrated=False):
        filtered_bricks = filter(lambda b: b.is_integrated == is_integrated, self.bricks)
        return list(filtered_bricks)

    @field_validator("bricks", mode="before")
    def validate_bricks(value):
        bricks = [sb if isinstance(sb, ServiceBrick) else ServiceBrick(**sb) for sb in value]
        return bricks

    @field_validator("land_use", mode="before")
    def validate_land_use(value):
        land_uses = [lu if isinstance(lu, LandUse) else LandUse[lu.upper()] for lu in value]
        return land_uses

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
