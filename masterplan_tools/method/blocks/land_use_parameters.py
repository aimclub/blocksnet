"""
Class holding parameters for land use filter parameters is defined here.
"""
import geopandas as gpd
from pydantic import BaseModel, Field, field_validator
from masterplan_tools.models.geojson import GeoJSON


class LandUseFeatureProperties(BaseModel):
    id: int | None = None


class LandUseParameters(BaseModel):
    """
    Parameters used for land use filter
    """

    landuse: GeoJSON[LandUseFeatureProperties]
    """basic landuse geometries"""
    no_development: GeoJSON[LandUseFeatureProperties]
    """territories with restricted development"""

    @field_validator("landuse", "no_development", mode="before")
    def validate_fields(value):
        if isinstance(value, gpd.GeoDataFrame):
            return GeoJSON[LandUseFeatureProperties].from_gdf(value)
        return value
