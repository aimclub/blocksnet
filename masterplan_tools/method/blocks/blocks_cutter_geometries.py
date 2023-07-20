"""
Class holding geometries used by block cutter is defined here.
"""
from typing import Optional
from pydantic import BaseModel
from masterplan_tools.models.geojson import GeoJSON


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
