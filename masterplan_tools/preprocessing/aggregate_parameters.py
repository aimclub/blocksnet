"""
Class holding geometries used by data getter is defined here.
"""
import geopandas as gpd
from pydantic import BaseModel, Field, field_validator
from ..models import PointGeoJSON


class AggregateBuildingsFeature(BaseModel):
    """
    The only feature required is a unique identifier.
    """

    population_balanced: int = Field(ge=0)
    building_area: float = Field(ge=0)
    living_area: float = Field(ge=0)
    storeys_count: int = Field(ge=0)
    is_living: bool
    living_area_pyatno: float = Field(ge=0)
    total_area: float = Field(ge=0)


class AggregateGreeningsFeature(BaseModel):
    """
    The only feature required is a unique identifier.
    """

    current_green_area: int = Field(ge=0)
    current_green_capacity: int = Field(ge=0)


class AggregateParkingsFeature(BaseModel):
    """
    The only feature required is a unique identifier.
    """

    current_parking_capacity: int = Field(ge=0)


class AggregateParameters(BaseModel):
    """
    Geometries used in blocks cutting process.
    """

    buildings: PointGeoJSON[AggregateBuildingsFeature]
    greenings: PointGeoJSON[AggregateGreeningsFeature]
    parkings: PointGeoJSON[AggregateParkingsFeature]

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
