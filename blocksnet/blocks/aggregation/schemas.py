import shapely
from pandera import Field, parser
from pandera.typing import Series

from ...utils.validation import GdfSchema


class BlocksSchema(GdfSchema):
    @classmethod
    def _geometry_types(cls):
        return {shapely.Polygon}
