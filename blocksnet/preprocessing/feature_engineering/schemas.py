import shapely
from ...utils.validation import GdfSchema


class BlocksSchema(GdfSchema):
    """Schema validating block geometries for feature engineering."""

    @classmethod
    def _geometry_types(cls):
        return {shapely.Polygon}
