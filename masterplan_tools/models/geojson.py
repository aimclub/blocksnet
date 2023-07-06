# pylint: disable=missing-module-docstring, no-name-in-module, too-few-public-methods, duplicate-code
"""
Geojson response model and its inner parts are defined here.
"""
from typing import Any, Generic, Iterable, Literal, TypeVar, Optional

import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, mapping
from pydantic import BaseModel, Field, create_model

FeaturePropertiesType = TypeVar("FeaturePropertiesType")  # pylint: disable=invalid-name


class Geometry(BaseModel):
    """Geometry representation for GeoJSON model."""

    type: Literal["Polygon", "MultiPolygon"]
    coordinates: list[Any] = []

    # @classmethod
    # def from_dict(cls, dict : dict[str, any]) -> 'Geometry':
    #   return cls(type=dict.type, coordinates=dict.coordinates)

    @classmethod
    def from_shapely_geometry(cls, geom: Polygon | MultiPolygon):
        tmp = mapping(geom)
        return cls(type=tmp["type"], coordinates=tmp["coordinates"])

    # def as_shapely_geometry(self) -> geom.Polygon | geom.MultiPolygon :
    #     """Return Shapely geometry object from the parsed geometry."""
    #     match self.type:
    #         case "Polygon":
    #             return geom.Polygon(self.coordinates[0])
    #         case "MultiPolygon":
    #             return geom.MultiPolygon(self.coordinates)


class Feature(BaseModel, Generic[FeaturePropertiesType]):
    """Feature representation for GeoJSON model."""

    geometry: Geometry
    properties: FeaturePropertiesType

    @classmethod
    def from_geoseries(cls, geoseries: gpd.GeoSeries) -> "Feature[FeaturePropertiesType]":
        """Construct Feature object from geoseries."""
        properties = geoseries.to_dict()
        del properties["geometry"]
        return cls(geometry=Geometry.from_shapely_geometry(geoseries.geometry), properties=properties)


class GeoJSON(BaseModel, Generic[FeaturePropertiesType]):
    """GeoJSON model representation."""

    features: list[Feature[FeaturePropertiesType]]

    @classmethod
    def from_gdf(cls, gdf: gpd.GeoDataFrame) -> "GeoJSON[FeaturePropertiesType]":
        """Construct GeoJSON model from geopandas GeoDataFrame."""
        features = list(gdf.apply(lambda row: Feature[FeaturePropertiesType].from_geoseries(row), axis=1))
        return cls(features=features)
