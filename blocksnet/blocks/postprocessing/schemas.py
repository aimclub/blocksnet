from shapely.geometry.base import BaseGeometry
from blocksnet.utils.validation import GdfSchema


class BlocksSchema(GdfSchema):
    """Schema accepting any geometry type produced during postprocessing."""

    @classmethod
    def _geometry_types(cls) -> set[type[BaseGeometry]]:
        return {BaseGeometry}
