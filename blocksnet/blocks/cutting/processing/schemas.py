import shapely
from ....common.validation import GdfSchema


class BoundariesSchema(GdfSchema):
    @classmethod
    def _geometry_types(cls):
        return {shapely.Polygon, shapely.MultiPolygon}


class LineObjectsSchema(GdfSchema):
    @classmethod
    def _geometry_types(cls):
        return {shapely.LineString}


class PolygonObjectsSchema(GdfSchema):
    @classmethod
    def _geometry_types(cls):
        return {shapely.Polygon}
