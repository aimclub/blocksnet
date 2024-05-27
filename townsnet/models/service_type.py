import math
from enum import Enum
import pandas as pd
from pydantic import BaseModel, Field, field_validator
from ..utils import SQUARE_METERS_IN_HECTARE

class ServiceInfrastructure(Enum):
    EDUCATION='education'
    HEALTHCARE='healthcare'
    COMMERCE='commerce'
    CATERING='catering'
    LEISURE='leisure'
    RECREATION='recreation'
    SPORT='sport'
    SERVICE='service'
    TRANSPORT='transport'
    SAFENESS='safeness'

class ServiceCategory(Enum):
    BASIC='basic'
    ADVANCED='advanced'
    COMFORT='comfort'

# class ServiceBrick(BaseModel):
#     """Typical service type brick or chunk, that is used in development"""

#     capacity: int
#     """How many people in need can be supported by a service"""
#     area: float
#     """Area in hectares"""
#     is_integrated: bool
#     """Is integrated within the building"""
#     parking_area: float

#     @property
#     def sq_m_area(self):
#         """Self area in square meters"""
#         return self.area * SQUARE_METERS_IN_HECTARE


class ServiceType(BaseModel):
    """Represents service type entity, such as schools and its parameters overall"""

    category : ServiceCategory
    infrastructure : ServiceInfrastructure
    name: str
    name_ru: str
    weight: float = Field(gt=0)
    accessibility: int = Field(gt=0)
    demand: int = Field(gt=0)
    osm_tags : dict
    # bricks: list[ServiceBrick] = []

    # @field_validator("bricks", mode="before")
    # def validate_bricks(value):
    #     bricks = [sb if isinstance(sb, ServiceBrick) else ServiceBrick(**sb) for sb in value]
    #     return bricks

    def calculate_in_need(self, population: int) -> int:
        """Calculate how many people in the given population are in need by this service type"""
        return math.ceil(population / 1000 * self.demand)

    def to_dict(self):
        return {
            'category': self.category.name,
            'infrastructure': self.infrastructure.name,
            'name': self.name,
            'name_ru': self.name_ru,
            'weight': self.weight,
            'accessibility': self.accessibility,
            'demand': self.demand
        }
    
    @staticmethod
    def to_df(iterable):
        if isinstance(iterable, dict):
            iterable = iterable.values()
        return pd.DataFrame([st.to_dict() for st in iterable])

    def __hash__(self):
        """Make service type hashable to use it as a key"""
        return hash(self.name)

    def __str__(self):
        return str.join('\n', [f'{key.ljust(15)}: {value}' for key, value in self.to_dict().items()])
