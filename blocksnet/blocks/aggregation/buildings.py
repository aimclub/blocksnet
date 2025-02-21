from pandera.typing import Series
from pandera import Field
from ...utils.validation import GdfSchema


class AreaAccessibilityBlocksSchema(GdfSchema):
    site_area: Series[float] = Field(ge=0)
