import geopandas as gpd
from shapely import Polygon
from pydantic import BaseModel, Field, InstanceOf


class Block(BaseModel):
    id: int
    geometry: InstanceOf[Polygon]
    population: int = Field(ge=0)
    floors: float = Field(ge=0)
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
    def from_gdf(cls, gdf: gpd.GeoDataFrame) -> "list[Block]":
        return (
            gdf.rename(
                columns={
                    "block_id": "id",
                    "current_population": "population",
                    "current_living_area": "living_area",
                    "current_green_capacity": "green_capacity",
                    "current_green_area": "green_area",
                    "current_parking_capacity": "parking_capacity",
                    "current_industrial_area": "industrial_area",
                },
                inplace=False,
            )
            .apply(lambda x: cls(**x.to_dict()), axis=1)
            .to_list()
        )

    def __hash__(self):
        return hash(self.id)
