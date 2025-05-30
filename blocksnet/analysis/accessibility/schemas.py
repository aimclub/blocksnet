from pandera import Field
from pandera.typing import Series

from ...utils.validation import DfSchema


class AreaAccessibilityBlocksSchema(DfSchema):
    site_area: Series[float] = Field(ge=0)
