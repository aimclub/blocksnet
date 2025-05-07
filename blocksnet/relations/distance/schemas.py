import shapely
from pandera.typing import Series
from pandera import Field, parser
from ...utils.validation import GdfSchema


class BlocksSchema(GdfSchema):
    @classmethod
    def _geometry_types(cls):
        return {shapely.Point}

    @parser("geometry")
    @classmethod
    def centrify(cls, geometry):
        return geometry.apply(lambda g: g.representative_point())
