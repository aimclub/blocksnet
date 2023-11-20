from pydantic import BaseModel, Field
import math


class ServiceType(BaseModel):
    """Represents service type entity, such as schools and its parameters overall"""

    name: str
    accessibility: int = Field(gt=0)
    demand: int = Field(gt=0)

    def calculate_in_need(self, population: int) -> int:
        """Calculate how many people in the given population are in need by this service type"""
        return math.ceil(population / 1000 * self.demand)

    def __hash__(self):
        """Make service type hashable to use it as a key"""
        return hash(self.name)

    def __str__(self):
        return f"{self.name} : {self.accessibility} min, {self.demand}/1000 population"
