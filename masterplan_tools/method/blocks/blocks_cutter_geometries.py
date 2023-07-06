from pydantic import BaseModel, FieldSerializationInfo, validator
from typing import Literal
from masterplan_tools.models.geojson import GeoJSON
import geopandas as gpd


class BlocksCutterFeatureType(BaseModel):
    id: int | None


class BlocksCutterGeometries(BaseModel):
    """
    Geometries used in blocks cutting process
    """

    city: GeoJSON[BlocksCutterFeatureType]
    water: GeoJSON[BlocksCutterFeatureType]
    roads: GeoJSON[BlocksCutterFeatureType]
    railways: GeoJSON[BlocksCutterFeatureType]
    nature: GeoJSON[BlocksCutterFeatureType]
