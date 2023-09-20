import geopandas as gpd
from shapely import Polygon
from pydantic import BaseModel, Field, InstanceOf


class Block(BaseModel):
    id: int
    geometry: InstanceOf[Polygon]
    population: int = Field(ge=0)
    floors: int = Field(ge=0)
    area: float = Field(ge=0)
    living_area: float = Field(ge=0)
    green_area: float = Field(ge=0)
    industrial_area: float = Field(ge=0)
    green_capacity: int = Field(ge=0)
    parking_capacity: int = Field(ge=0)

    @property
    def is_living(self):
        return self.population > 0

    @classmethod
    def from_gdf(gdf):

        return

    def __hash__(self):
        return hash(self.id)
