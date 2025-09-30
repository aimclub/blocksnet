import shapely
from pandera.typing import Series
from pandera import Field, parser
from ...utils.validation import GdfSchema


class BlocksSchema(GdfSchema):
    """Schema validating block geometries for aggregation."""

    @classmethod
    def _geometry_types(cls):
        return {shapely.Polygon}
