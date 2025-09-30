import shapely
from pandera.typing import Series
from ...utils.validation import GdfSchema


class BlocksSchema(GdfSchema):
    """Schema validating polygon blocks for land-use assignment."""

    @classmethod
    def _geometry_types(cls):
        return {shapely.Polygon}


class FunctionalZonesSchema(GdfSchema):
    """Schema describing functional zones used during land-use assignment."""

    functional_zone: Series[str]

    @classmethod
    def _geometry_types(cls):
        return {shapely.Polygon, shapely.MultiPolygon}
