"""
Geojson response model and its inner parts are defined here.
"""
from typing import Any, Generic, Literal, TypeVar

import geopandas as gpd
import pandas as pd
from pydantic import BaseModel
from shapely.geometry import mapping
from shapely.geometry.base import BaseGeometry


class Geometry(BaseModel):
    """Geometry representation for GeoJSON model."""

    type: Literal["Polygon", "MultiPolygon"]
    coordinates: list[Any] = []

    # @classmethod
    # def from_dict(cls, dict : dict[str, any]) -> 'Geometry':
    #   return cls(type=dict.type, coordinates=dict.coordinates)

    @classmethod
    def from_shapely_geometry(cls, geom: BaseGeometry) -> "Geometry":
        """Construct geometry from shapely BaseGeometry"""
        tmp = mapping(geom)
        return cls(type=tmp["type"], coordinates=tmp["coordinates"])

    # def as_shapely_geometry(self) -> geom.Polygon | geom.MultiPolygon :
    #     """Return Shapely geometry object from the parsed geometry."""
    #     match self.type:
    #         case "Polygon":
    #             return geom.Polygon(self.coordinates[0])
    #         case "MultiPolygon":
    #             return geom.MultiPolygon(self.coordinates)


_FeaturePropertiesType = TypeVar("_FeaturePropertiesType")  # pylint: disable=invalid-name


class Feature(BaseModel, Generic[_FeaturePropertiesType]):
    """Feature representation for GeoJSON model."""

    geometry: Geometry
    properties: _FeaturePropertiesType

    @classmethod
    def from_geoseries(cls, geoseries: pd.Series) -> "Feature[_FeaturePropertiesType]":
        """Construct Feature object from geoseries."""
        properties = geoseries.to_dict()
        del properties["geometry"]
        return cls(geometry=Geometry.from_shapely_geometry(geoseries.geometry), properties=properties)


_GeoJSONFeatureType = TypeVar("_GeoJSONFeatureType")  # pylint: disable=invalid-name


class GeoJSON(BaseModel, Generic[_GeoJSONFeatureType]):
    """GeoJSON model representation."""

    features: list[Feature[_GeoJSONFeatureType]]

    @classmethod
    def from_gdf(cls, gdf: gpd.GeoDataFrame) -> "GeoJSON[_GeoJSONFeatureType]":
        """Construct GeoJSON model from geopandas GeoDataFrame."""
        runtime_feature_type = cls.__pydantic_generic_metadata__["args"][0]
        features = gdf.apply(Feature[runtime_feature_type].from_geoseries, axis=1).to_list()
        return cls(features=features)
