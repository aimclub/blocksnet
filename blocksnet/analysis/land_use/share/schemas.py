from pandera import Field
from pandera.typing import Series
from ....utils.validation import LandUseSchema


class BlocksSchema(LandUseSchema):
    """BlocksSchema class.

    """
    site_area: Series[float] = Field(ge=0)
