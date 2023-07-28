"""
Class holding parameters for blocks clustering.
"""
import geopandas as gpd
from pydantic import BaseModel, field_validator
from masterplan_tools.models.geojson import GeoJSON


class ClusteringFeatureProperties(BaseModel):
    id: int | None = None


class ClusteringParameters(BaseModel):
    """
    Parameters used for blocks clustering
    """

    buildings: GeoJSON[ClusteringFeatureProperties]
    """buildings geometries that are used for clustering inside of blocks"""

    @field_validator("buildings", mode="before")
    def validate_fields(value):
        if isinstance(value, gpd.GeoDataFrame):
            return GeoJSON[ClusteringFeatureProperties].from_gdf(value)
        return value
