import shapely

from ...utils.validation import GdfSchema


class BlocksSchema(GdfSchema):
    @classmethod
    def _geometry_types(cls):
        return {shapely.Polygon}
