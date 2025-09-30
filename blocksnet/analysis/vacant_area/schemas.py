import shapely
from ...utils.validation import GdfSchema


class BlocksSchema(GdfSchema):
    """Schema ensuring vacant area inputs contain polygon geometries."""
    @classmethod
    def _geometry_types(cls):
        return {shapely.Polygon}
