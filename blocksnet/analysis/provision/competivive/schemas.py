from pandera.typing import Series
from pandera import Field
from ....utils.validation import DfSchema


class BlocksSchema(DfSchema):
    """Schema validating demand and capacity inputs for competitive provision."""

    demand: Series[int] = Field(ge=0)
    capacity: Series[int] = Field(ge=0)
