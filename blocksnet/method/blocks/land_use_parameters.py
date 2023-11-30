"""
Class holding parameters for land use filter parameters is defined here.
"""
import geopandas as gpd
from pydantic import BaseModel, Field, field_validator
from shapely import MultiPolygon, Polygon

from ...models import BaseRow, GeoDataFrame


class LandUseRow(BaseRow):
    """
    Landuse columns
    """

    geometry: Polygon | MultiPolygon
    """
    Unique identifier
    """


class LandUseParameters(BaseModel):
    """
    Parameters used for land use filter
    """

    landuse: GeoDataFrame[LandUseRow]
    """Basic landuse geometries"""
    no_development: GeoDataFrame[LandUseRow]
    """Territories with restricted development"""
    buildings: GeoDataFrame[LandUseRow] = None
    """Buildings geometries that are used for clustering inside of blocks"""

    @field_validator("landuse", "no_development", "buildings", mode="before")
    def validate_fields(value):
        if isinstance(value, gpd.GeoDataFrame):
            return GeoDataFrame[LandUseRow](value)
        return value
