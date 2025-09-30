from shapely.geometry.base import BaseGeometry
from blocksnet.utils.validation import GdfSchema


class BlocksSchema(GdfSchema):
    @classmethod
    """BlocksSchema class.

    """
    def _geometry_types(cls) -> set[type[BaseGeometry]]:
        """Geometry types.

        Returns
        -------
        set[type[BaseGeometry]]
            Description.

        """
        return {BaseGeometry}
