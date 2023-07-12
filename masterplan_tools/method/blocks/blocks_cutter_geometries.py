"""
Class holding geometries used by block cutter is defined here.
"""
from typing import Optional
from pydantic import BaseModel
from masterplan_tools.models.geojson import GeoJSON


class BlocksCutterFeatureType(BaseModel):
    """
    The only feature required is a unique identifier.
    """

    id: Optional[int] = None


class BlocksCutterGeometries(BaseModel):
    """
    Geometries used in blocks cutting process.
    """

    city: GeoJSON[BlocksCutterFeatureType]
    water: GeoJSON[BlocksCutterFeatureType]
    roads: GeoJSON[BlocksCutterFeatureType]
    railways: GeoJSON[BlocksCutterFeatureType]
    nature: GeoJSON[BlocksCutterFeatureType]
