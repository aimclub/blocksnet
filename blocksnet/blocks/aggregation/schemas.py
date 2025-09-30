import shapely
from pandera.typing import Series
from pandera import Field, parser
from ...utils.validation import GdfSchema


class BlocksSchema(GdfSchema):
    @classmethod
    """BlocksSchema class.

    """
    def _geometry_types(cls):
        """Geometry types.

        """
        return {shapely.Polygon}
