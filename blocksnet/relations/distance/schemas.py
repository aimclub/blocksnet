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
        return {shapely.Point}

    @parser("geometry")
    @classmethod
    def centrify(cls, geometry):
        """Centrify.

        Parameters
        ----------
        geometry : Any
            Description.

        """
        return geometry.apply(lambda g: g.representative_point())
