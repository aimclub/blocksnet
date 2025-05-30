from pandera import Field
from pandera.typing import Series

from ...utils.validation import LandUseSchema


class BlocksSchema(LandUseSchema):
    area: Series[float] = Field(ge=0)
    # aspect_ratio : Series[float] = Field(ge=0)
