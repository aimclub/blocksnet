"""
Class holding parameters for base blocks cut is defined here.
"""
import geopandas as gpd
from pydantic import BaseModel, Field, field_validator
from masterplan_tools.models.geojson import PolygonGeoJSON


class CutFeatureProperties(BaseModel):
    id: int | None = None


class CutParameters(BaseModel):
    """
    Parameters used for base blocks cut process
    """

    roads_buffer: int = Field(5, ge=0)
    """roads geometry buffer in meters used to fill dead ends inside blocks, should be the same size as roads width"""
    city: PolygonGeoJSON[CutFeatureProperties]
    water: PolygonGeoJSON[CutFeatureProperties]
    roads: PolygonGeoJSON[CutFeatureProperties]
    railways: PolygonGeoJSON[CutFeatureProperties]

    @field_validator("city", "water", "roads", "railways", mode="before")
    def validate_fields(value):
        if isinstance(value, gpd.GeoDataFrame):
            return PolygonGeoJSON[CutFeatureProperties].from_gdf(value)
        return value
