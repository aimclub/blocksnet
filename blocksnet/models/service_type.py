from pydantic import BaseModel, Field
import math


class ServiceType(BaseModel):
    """Represents service type entity, such as schools and its parameters overall"""

    name: str
    accessibility: int = Field(gt=0)
    demand: int = Field(gt=0)
    buffer: int = Field(ge=0, default=0)

    def calculate_in_need(self, population: int) -> int:
        return math.ceil(population / 1000 * self.demand)

    def __hash__(self):
        return hash(self.name)
