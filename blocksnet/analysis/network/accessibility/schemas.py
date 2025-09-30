"""Validation schemas for accessibility indicators."""

from pandera.typing import Series
from pandera import Field

from blocksnet.utils.validation import DfSchema, LandUseSchema


class AreaAccessibilityBlocksSchema(DfSchema):
    """Validate block areas used for area-weighted accessibility calculations."""

    site_area: Series[float] = Field(ge=0)


class LandUseAccessibilityBlocksSchema(LandUseSchema):
    """Validate block land-use assignments for land-use accessibility statistics."""

    pass
