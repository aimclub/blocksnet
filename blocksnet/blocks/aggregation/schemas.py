import shapely
from pandera.typing import Series
from pandera import Field
from ...utils.validation import GdfSchema


class BlocksSchema(GdfSchema):
    @classmethod
    def _geometry_types(cls):
        return {shapely.Polygon}


class ObjectsSchema(GdfSchema):
    class Config:
        add_missing_columns = True
        coerce = True

    @classmethod
    def _geometry_types(cls):
        return {
            shapely.Polygon,
            shapely.MultiPolygon,
            shapely.Point,
            shapely.MultiPoint,
            shapely.LineString,
            shapely.MultiLineString,
        }
