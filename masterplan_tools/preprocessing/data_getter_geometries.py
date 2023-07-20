"""
Class holding geometries used by data getter is defined here.
"""
from typing import Optional
from pydantic import BaseModel
from masterplan_tools.models.geojson import GeoJSON


class DataGetterBuildingsFeature(BaseModel):
    """
    The only feature required is a unique identifier.
    """

    id: Optional[int] = None


class DataGetterGreeningsFeature(BaseModel):
    """
    The only feature required is a unique identifier.
    """

    id: Optional[int] = None


class DataGetterParkingsFeature(BaseModel):
    """
    The only feature required is a unique identifier.
    """

    id: Optional[int] = None


class DataGetterGeometries(BaseModel):
    """
    Geometries used in blocks cutting process.
    """

    buildings: GeoJSON[DataGetterBuildingsFeature]
    greenings: GeoJSON[DataGetterGreeningsFeature]
    parkings: GeoJSON[DataGetterParkingsFeature]
