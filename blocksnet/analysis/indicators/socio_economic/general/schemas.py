from pandera import Field
from pandera.typing import Series
from blocksnet.utils.validation import LandUseSchema


class BlocksSchema(LandUseSchema):
    site_area: Series[float] = Field(ge=0)
