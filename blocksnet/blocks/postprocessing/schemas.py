from shapely.geometry.base import BaseGeometry
from blocksnet.utils.validation import GdfSchema


class BlocksSchema(GdfSchema):
    @classmethod
    def _geometry_types(cls) -> set[type[BaseGeometry]]:
        return {BaseGeometry}
