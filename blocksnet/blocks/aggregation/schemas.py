import shapely
from pandera.typing import Series
from pandera import Field
from ...utils.validation import GdfSchema


class BlocksSchema(GdfSchema):
    @classmethod
    def _geometry_types(cls):
        return {shapely.Polygon}


class BuildingsSchema(GdfSchema):
    @classmethod
    def _geometry_types(cls):
        return {shapely.Polygon, shapely.MultiPolygon, shapely.Point}


class ParameteredBuildingsSchema(BuildingsSchema):
    build_floor_area: Series[float] = Field(ge=0)
    living_area: Series[float] = Field(ge=0)
    non_living_area: Series[float] = Field(ge=0)
    footprint_area: Series[float] = Field(ge=0)
    number_of_floors: Series[int] = Field(ge=1)


class PopulatedBuildingSchema(BuildingsSchema):
    population: Series[int] = Field(ge=0)
