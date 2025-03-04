import shapely
from pandera.typing import Series
from ...common.validation import GdfSchema


class BlocksSchema(GdfSchema):
    @classmethod
    def _geometry_types(cls):
        return {shapely.Polygon}


class FunctionalZonesSchema(GdfSchema):

    functional_zone: Series[str]

    @classmethod
    def _geometry_types(cls):
        return {shapely.Polygon, shapely.MultiPolygon}
