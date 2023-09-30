"""
Class holding parameters for base blocks cut is defined here.
"""
import geopandas as gpd
from pydantic import BaseModel, Field, field_validator
from blocksnet.models.geojson import PolygonGeoJSON


class CutFeatureProperties(BaseModel):
    """
    Landuse features properties
    """

    id: int | None = None
    """
    Unique identifier
    """


class CutParameters(BaseModel):
    """
    Parameters used for base blocks cut process
    """

    roads_buffer: int = Field(5, ge=0)
    """roads geometry buffer in meters used to fill dead ends inside blocks, should be the same size as roads width"""
    city: PolygonGeoJSON[CutFeatureProperties]
    """city boundaries geometry"""
    water: PolygonGeoJSON[CutFeatureProperties]
    """water objects geometries"""
    roads: PolygonGeoJSON[CutFeatureProperties]
    """road network geometries"""
    railways: PolygonGeoJSON[CutFeatureProperties]
    """railways network geometries"""

    @field_validator("city", "water", "roads", "railways", mode="before")
    def validate_fields(value):
        if isinstance(value, gpd.GeoDataFrame):
            return PolygonGeoJSON[CutFeatureProperties].from_gdf(value)
        return value
