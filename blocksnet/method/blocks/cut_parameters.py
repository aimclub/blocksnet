"""
Class holding parameters for base blocks cut is defined here.
"""
import geopandas as gpd
from pydantic import BaseModel, Field, field_validator
from ...models import GeoDataFrame, BaseRow


class CutRow(BaseRow):
    """
    Cut geometries columns
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
    city: GeoDataFrame[CutRow]
    """city boundaries geometry"""
    water: GeoDataFrame[CutRow]
    """water objects geometries"""
    roads: GeoDataFrame[CutRow]
    """road network geometries"""
    railways: GeoDataFrame[CutRow]
    """railways network geometries"""

    @field_validator("city", "water", "roads", "railways", mode="before")
    def validate_fields(value):
        if isinstance(value, gpd.GeoDataFrame):
            return GeoDataFrame[CutRow](value)
        return value
