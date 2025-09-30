import shapely
from ....utils.validation import GdfSchema


class RoadsSchema(GdfSchema):
    """Schema validating linear road geometries for preprocessing."""

    @classmethod
    def _geometry_types(cls):
        return {shapely.LineString, shapely.MultiLineString}


class RailwaysSchema(GdfSchema):
    """Schema validating railway geometries prior to block cutting."""

    @classmethod
    def _geometry_types(cls):
        return {shapely.LineString, shapely.MultiLineString}


class WaterSchema(GdfSchema):
    """Schema validating water geometries used in preprocessing."""

    @classmethod
    def _geometry_types(cls):
        return {shapely.LineString, shapely.MultiLineString, shapely.Polygon, shapely.MultiPolygon}
