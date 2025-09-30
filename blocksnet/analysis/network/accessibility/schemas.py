from pandera.typing import Series
from pandera import Field
from blocksnet.utils.validation import DfSchema, LandUseSchema


class AreaAccessibilityBlocksSchema(DfSchema):
    """Schema ensuring area-weighted accessibility inputs are valid.

    Requires ``site_area`` values for each block to be present and
    non-negative.
    """

    site_area: Series[float] = Field(ge=0)


class LandUseAccessibilityBlocksSchema(LandUseSchema):
    """Schema validating land-use annotated block data for accessibility."""

    pass
