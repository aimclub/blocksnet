"""
Class holding geometries used by data getter is defined here.
"""
import geopandas as gpd
from pydantic import BaseModel, Field, field_validator
from ..models import GeoDataFrame, BaseRow


class AggregateBuildingsRow(BaseRow):
    """
    Buildings columns
    """

    population_balanced: int = Field(ge=0)
    """Total population of the building"""
    building_area: float = Field(ge=0)
    """Building area (in square meters)"""
    living_area: float = Field(ge=0)
    """Living area (in square meters)"""
    storeys_count: int = Field(ge=0)
    """Storeys count of the building"""
    is_living: bool
    """Is building living"""
    living_area_pyatno: float = Field(ge=0)
    """Living area pyatno (in square meters)"""
    total_area: float = Field(ge=0)
    """Total building area (in square meters)"""


class AggregateGreeningsRow(BaseRow):
    """
    Greenings columns
    """

    current_green_area: int = Field(ge=0)
    """Greening area (in square meters)"""
    current_green_capacity: int = Field(ge=0)
    """Total greening capacity (in units)"""


class AggregateParkingsRow(BaseRow):
    """
    Parkings columns
    """

    current_parking_capacity: int = Field(ge=0)
    """Total parking capacity (in units)"""


class AggregateParameters(BaseModel):
    """
    Geometries used in parameters aggregation process.
    """

    buildings: GeoDataFrame[AggregateBuildingsRow]
    """Buildings geometries"""
    greenings: GeoDataFrame[AggregateGreeningsRow]
    """Green areas geometries"""
    parkings: GeoDataFrame[AggregateParkingsRow]
    """Parkings geometries"""

    @field_validator("buildings", mode="before")
    def validate_buildings(value):
        if isinstance(value, gpd.GeoDataFrame):
            return GeoDataFrame[AggregateBuildingsRow](value)
        return value

    @field_validator("greenings", mode="before")
    def validate_greenings(value):
        if isinstance(value, gpd.GeoDataFrame):
            return GeoDataFrame[AggregateGreeningsRow](value)
        return value

    @field_validator("parkings", mode="before")
    def validate_parkings(value):
        if isinstance(value, gpd.GeoDataFrame):
            return GeoDataFrame[AggregateParkingsRow](value)
        return value
