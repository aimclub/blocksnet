from pandera.typing import Series
from pandera import Field
from blocksnet.utils.validation import DfSchema, LandUseSchema


class AreaAccessibilityBlocksSchema(DfSchema):
    """AreaAccessibilityBlocksSchema class.

    """
    site_area: Series[float] = Field(ge=0)


class LandUseAccessibilityBlocksSchema(LandUseSchema):
    """LandUseAccessibilityBlocksSchema class.

    """
    pass
