import shapely
from pandera.typing import Series
from ...utils.validation import GdfSchema


class BlocksSchema(GdfSchema):
    @classmethod
    """BlocksSchema class.

    """
    def _geometry_types(cls):
        """Geometry types.

        """
        return {shapely.Polygon}


class FunctionalZonesSchema(GdfSchema):

    """FunctionalZonesSchema class.

    """
    functional_zone: Series[str]

    @classmethod
    def _geometry_types(cls):
        """Geometry types.

        """
        return {shapely.Polygon, shapely.MultiPolygon}
