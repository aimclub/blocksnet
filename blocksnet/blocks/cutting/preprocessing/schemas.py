import shapely
from ....utils.validation import GdfSchema


class RoadsSchema(GdfSchema):
    @classmethod
    def _geometry_types(cls):
        return {shapely.LineString, shapely.MultiLineString}


class RailwaysSchema(GdfSchema):
    @classmethod
    def _geometry_types(cls):
        return {shapely.LineString, shapely.MultiLineString}


class WaterSchema(GdfSchema):
    @classmethod
    def _geometry_types(cls):
        return {shapely.LineString, shapely.MultiLineString, shapely.Polygon, shapely.MultiPolygon}
