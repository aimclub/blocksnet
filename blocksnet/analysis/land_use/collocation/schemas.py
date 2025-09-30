from pandera import Field
from pandera.typing import Series
from ....utils.validation import LandUseSchema


class BlocksSchema(LandUseSchema):
    """Schema validating land-use blocks with available site areas."""

    site_area: Series[float] = Field(ge=0)
