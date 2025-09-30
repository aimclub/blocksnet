from pandera import Field
from pandera.typing import Series
from shapely.geometry import LineString, MultiLineString
from shapely.geometry.base import BaseGeometry
from blocksnet.utils.validation import DfSchema, GdfSchema, LandUseSchema


class BlocksAreaSchema(DfSchema):
    """Schema validating block area measurements for transport indicators."""

    site_area: Series[float] = Field(ge=0)


class BlocksAccessibilitySchema(LandUseSchema):
    """Schema ensuring accessibility calculations have service counts."""

    count: Series[int] = Field(ge=0)


class NetworkSchema(GdfSchema):
    """Schema restricting transport networks to linear geometries."""
    @classmethod
    def _geometry_types(cls) -> set[type[BaseGeometry]]:
        return {LineString, MultiLineString}
