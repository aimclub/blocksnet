from pandera.typing import Series
from pandera import Field
from blocksnet.utils.validation import DfSchema, LandUseSchema


class AreaAccessibilityBlocksSchema(DfSchema):
    site_area: Series[float] = Field(ge=0)


class LandUseAccessibilityBlocksSchema(LandUseSchema):
    pass
