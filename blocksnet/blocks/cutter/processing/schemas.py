import shapely
import pandas as pd
from pandera import dataframe_parser
from ....utils.validation import GdfSchema


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
