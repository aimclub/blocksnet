"""
Geojson response model and its inner parts are defined here.
"""
from typing import Any, Generic, Literal, TypeVar

import geopandas as gpd
from pydantic import BaseModel
from shapely.geometry import mapping
from shapely.geometry.base import BaseGeometry


class Geometry(BaseModel):
    """Geometry representation for GeoJSON model"""

    type: Literal["Polygon", "MultiPolygon", "Point"]
    """Geometry type"""
    coordinates: list[Any] = []
    """Geometry coordinates list"""

    # @classmethod
    # def from_dict(cls, dict : dict[str, any]) -> 'Geometry':
    #   return cls(type=dict.type, coordinates=dict.coordinates)

    @classmethod
    def from_shapely_geometry(cls, geom: BaseGeometry) -> "Geometry":
        """Construct geometry from shapely BaseGeometry"""
        tmp = mapping(geom)
        return cls(type=tmp["type"], coordinates=tmp["coordinates"])

    # @staticmethod
    # def _coordinates_to_polygon(coordinates: list[any]) -> Polygon:
    #     return Polygon(coordinates)

    def to_dict(self) -> dict["str", any]:
        """Generate dict for the Feature"""
        return {"type": self.type, "coordinates": self.coordinates}


class PointGeometry(Geometry):
    """Point geometry representation for GeoJSON model"""

    type: Literal["Point"]
    """Point geometry type"""


class PolygonGeometry(Geometry):
    """Polygon and multipolygon geometries representation for GeoJSON model"""

    type: Literal["Polygon", "MultiPolygon"]
    """Polygon geometry type"""


_FeaturePropertiesType = TypeVar("_FeaturePropertiesType")  # pylint: disable=invalid-name


class Feature(BaseModel, Generic[_FeaturePropertiesType]):
    """Feature representation for GeoJSON model."""

    geometry: Geometry
    """Feature geometry"""
    properties: _FeaturePropertiesType
    """Feature properties"""

    @staticmethod
    def _geometry_from_shapely(geoseries):
        return Geometry.from_shapely_geometry(geoseries.geometry)

    @classmethod
    def from_geoseries(cls, geoseries: gpd.GeoSeries) -> "Feature[_FeaturePropertiesType]":
        """Construct Feature object from geoseries."""
        properties = geoseries.to_dict()
        del properties["geometry"]
        return cls(geometry=cls._geometry_from_shapely(geoseries), properties=properties)

    def to_dict(self) -> dict["str", any]:
        """Generate dict for the Feature"""
        dict = {"type": "Feature", "geometry": self.geometry.to_dict(), "properties": {}}
        for field_name in self.properties.model_fields.keys():
            dict["properties"][field_name] = self.properties.__getattribute__(field_name)
        return dict


class PointFeature(Feature, Generic[_FeaturePropertiesType]):
    """Point feature representation for GeoJSON model"""

    geometry: PointGeometry
    """Feature geometry"""

    @staticmethod
    def _geometry_from_shapely(geoseries):
        return PointGeometry.from_shapely_geometry(geoseries.geometry)


class PolygonFeature(Feature, Generic[_FeaturePropertiesType]):
    """Polygon and multipolygon features representation for GeoJSON model"""

    geometry: PolygonGeometry
    """Feature geometry"""

    @staticmethod
    def _geometry_from_shapely(geoseries):
        """Generate Geometry from Shapely"""
        return PolygonGeometry.from_shapely_geometry(geoseries.geometry)


_GeoJSONFeatureType = TypeVar("_GeoJSONFeatureType")  # pylint: disable=invalid-name


class GeoJSON(BaseModel, Generic[_GeoJSONFeatureType]):
    """GeoJSON model representation."""

    epsg: int
    """EPSG value"""
    features: list[Feature[_GeoJSONFeatureType]]
    """GeoJSON features list"""

    @staticmethod
    def _gdf_to_features(gdf, runtime_feature_type):
        """Generate Features from GeoDataFrame"""
        return gdf.apply(Feature[runtime_feature_type].from_geoseries, axis=1).to_list()

    @classmethod
    def from_gdf(cls, gdf: gpd.GeoDataFrame) -> "GeoJSON[_GeoJSONFeatureType]":
        """Construct GeoJSON model from geopandas GeoDataFrame."""
        runtime_feature_type = cls.__pydantic_generic_metadata__["args"][0]
        features = cls._gdf_to_features(gdf, runtime_feature_type)
        return cls(features=features, epsg=gdf.crs.to_epsg())

    def to_gdf(self) -> gpd.GeoDataFrame:
        """Generate GeoDataFrame for the object"""
        gdf = gpd.GeoDataFrame.from_features(map(lambda feature: feature.to_dict(), self.features))
        gdf.set_crs(epsg=self.epsg, inplace=True)
        return gdf


class PointGeoJSON(GeoJSON, Generic[_GeoJSONFeatureType]):
    """Class for representing GeoJSON, containing point features"""

    features: list[PointFeature[_GeoJSONFeatureType]]

    @staticmethod
    def _gdf_to_features(gdf, runtime_feature_type):
        """Generate Features from GeoDataFrame"""
        return gdf.apply(PointFeature[runtime_feature_type].from_geoseries, axis=1).to_list()


class PolygonGeoJSON(GeoJSON, Generic[_GeoJSONFeatureType]):
    """Class for representing GeoJSON, containing polygon and multipolygon features"""

    features: list[PolygonFeature[_GeoJSONFeatureType]]

    @staticmethod
    def _gdf_to_features(gdf, runtime_feature_type):
        """Generate Features from GeoDataFrame"""
        return gdf.apply(PolygonFeature[runtime_feature_type].from_geoseries, axis=1).to_list()
