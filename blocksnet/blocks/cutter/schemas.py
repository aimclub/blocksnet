import shapely
import pandas as pd
from pandera import dataframe_parser
from ...utils.validation import GdfSchema


class BoundariesSchema(GdfSchema):
    @classmethod
    def _geometry_types(cls):
        return {shapely.Polygon, shapely.MultiPolygon}


class RoadsSchema(GdfSchema):
    @classmethod
    def _geometry_types(cls):
        return {shapely.LineString}

    @dataframe_parser
    @classmethod
    def explode(cls, df: pd.DataFrame) -> pd.DataFrame:
        return df.explode("geometry", True)


class RailwaysSchema(GdfSchema):
    @classmethod
    def _geometry_types(cls):
        return {shapely.LineString}

    @dataframe_parser
    @classmethod
    def explode(cls, df: pd.DataFrame) -> pd.DataFrame:
        return df.explode("geometry", True)


class WaterSchema(GdfSchema):
    @classmethod
    def _geometry_types(cls):
        return {shapely.LineString, shapely.Polygon}

    @dataframe_parser
    @classmethod
    def explode(cls, df: pd.DataFrame) -> pd.DataFrame:
        return df.explode("geometry", True)


class LineObjectsSchema(GdfSchema):
    @classmethod
    def _geometry_types(cls):
        return {shapely.LineString}

    @dataframe_parser
    @classmethod
    def explode(cls, df: pd.DataFrame) -> pd.DataFrame:
        return df.explode("geometry", True)


class PolygonObjectsSchema(GdfSchema):
    @classmethod
    def _geometry_types(cls):
        return {shapely.Polygon}

    @dataframe_parser
    @classmethod
    def explode(cls, df: pd.DataFrame) -> pd.DataFrame:
        return df.explode("geometry", True)
