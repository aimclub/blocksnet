import shapely
from blocksnet.utils.validation import GdfSchema


class BlocksSchema(GdfSchema):
    @classmethod
    """BlocksSchema class.

    """
    def _geometry_types(cls) -> set[type[shapely.geometry.base.BaseGeometry]]:
        """Geometry types.

        Returns
        -------
        set[type[shapely.geometry.base.BaseGeometry]]
            Description.

        """
        return {shapely.Polygon}
