from pandera import Field
from pandera.typing import Series
from blocksnet.utils.validation import LandUseSchema


class BlocksSchema(LandUseSchema):
    """Schema validating land-use blocks for general socio-economic metrics."""

    site_area: Series[float] = Field(ge=0)
