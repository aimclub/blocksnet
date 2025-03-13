import shapely
from ....utils.validation import GdfSchema


class RoadsSchema(GdfSchema):
    @classmethod
    def _geometry_types(cls):
        return {shapely.LineString}


class RailwaysSchema(GdfSchema):
    @classmethod
    def _geometry_types(cls):
        return {shapely.LineString}


class WaterSchema(GdfSchema):
    @classmethod
    def _geometry_types(cls):
        return {shapely.LineString, shapely.Polygon}
