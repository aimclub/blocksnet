"""
Class holding geometries used by block cutter is defined here.
"""
from typing import Optional
from pydantic import BaseModel, field_validator
from masterplan_tools.models.geojson import GeoJSON
from geopandas import GeoDataFrame


class BlocksCutterFeature(BaseModel):
    """
    The only feature required is a unique identifier.
    """

    id: Optional[int] = None


class BlocksCutterGeometries(BaseModel):
    """
    Geometries used in blocks cutting process.
    """

    city: GeoJSON[BlocksCutterFeature]
    water: GeoJSON[BlocksCutterFeature]
    roads: GeoJSON[BlocksCutterFeature]
    railways: GeoJSON[BlocksCutterFeature]
    nature: GeoJSON[BlocksCutterFeature]
    no_development: GeoJSON[BlocksCutterFeature]
    landuse: GeoJSON[BlocksCutterFeature]

    @field_validator("*", mode="before")
    def validate_fields(value):
        if isinstance(value, GeoDataFrame):
            return GeoJSON[BlocksCutterFeature].from_gdf(value)
        return value
