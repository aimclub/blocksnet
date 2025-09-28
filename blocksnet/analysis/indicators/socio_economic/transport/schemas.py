from pandera import Field
from pandera.typing import Series
from shapely.geometry import LineString, MultiLineString
from shapely.geometry.base import BaseGeometry
from blocksnet.utils.validation import DfSchema, GdfSchema, LandUseSchema


class BlocksAreaSchema(DfSchema):
    site_area: Series[float] = Field(ge=0)


class BlocksAccessibilitySchema(LandUseSchema):
    count: Series[int] = Field(ge=0)


class NetworkSchema(GdfSchema):
    @classmethod
    def _geometry_types(cls) -> set[type[BaseGeometry]]:
        return {LineString, MultiLineString}
