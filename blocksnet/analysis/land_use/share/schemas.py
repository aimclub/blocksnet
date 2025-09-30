from pandera import Field
from pandera.typing import Series
from ....utils.validation import LandUseSchema


class BlocksSchema(LandUseSchema):
    """Schema validating land-use annotated blocks with site area."""

    site_area: Series[float] = Field(ge=0)
