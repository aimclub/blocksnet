from pandera.typing import Series
from pandera import Field
from ....utils.validation import DfSchema


class BlocksSchema(DfSchema):

    """BlocksSchema class.

    """
    demand: Series[int] = Field(ge=0)
    capacity: Series[int] = Field(ge=0)
