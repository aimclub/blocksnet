"""
Class holding geometries used by data getter is defined here.
"""
import geopandas as gpd
from pydantic import BaseModel, Field, field_validator
from ..models import PointGeoJSON


class AggregateBuildingsFeature(BaseModel):
    """
    Buildings features properties
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


class AggregateGreeningsFeature(BaseModel):
    """
    Greenings features properties
    """

    current_green_area: int = Field(ge=0)
    """Greening area (in square meters)"""
    current_green_capacity: int = Field(ge=0)
    """Total greening capacity (in units)"""


class AggregateParkingsFeature(BaseModel):
    """
    Parkings features properties
    """

    current_parking_capacity: int = Field(ge=0)
    """Total parking capacity (in units)"""


class AggregateParameters(BaseModel):
    """
    Geometries used in parameters aggregation process.
    """

    buildings: PointGeoJSON[AggregateBuildingsFeature]
    """Buildings geometries"""
    greenings: PointGeoJSON[AggregateGreeningsFeature]
    """Green areas geometries"""
    parkings: PointGeoJSON[AggregateParkingsFeature]
    """Parkings geometries"""

    @field_validator("buildings", mode="before")
    def validate_buildings(value):
        if isinstance(value, gpd.GeoDataFrame):
            return PointGeoJSON[AggregateBuildingsFeature].from_gdf(value)
        return value

    @field_validator("greenings", mode="before")
    def validate_greenings(value):
        if isinstance(value, gpd.GeoDataFrame):
            return PointGeoJSON[AggregateGreeningsFeature].from_gdf(value)
        return value

    @field_validator("parkings", mode="before")
    def validate_parkings(value):
        if isinstance(value, gpd.GeoDataFrame):
            return PointGeoJSON[AggregateParkingsFeature].from_gdf(value)
        return value
