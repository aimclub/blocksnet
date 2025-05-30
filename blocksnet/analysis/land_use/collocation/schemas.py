from pandera import Field
from pandera.typing import Series

from ....utils.validation import LandUseSchema


class BlocksSchema(LandUseSchema):
    site_area: Series[float] = Field(ge=0)
