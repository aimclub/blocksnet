import shapely
from blocksnet.utils.validation import GdfSchema


class BlocksSchema(GdfSchema):
    @classmethod
    def _geometry_types(cls) -> set[type[shapely.geometry.base.BaseGeometry]]:
        return {shapely.Polygon}
