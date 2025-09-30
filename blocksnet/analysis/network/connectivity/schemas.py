from pandera.typing import Series
from pandera import Field
from blocksnet.utils.validation import DfSchema


class BlocksAccessibilitySchema(DfSchema):
    """Schema enforcing the presence of a non-negative accessibility column."""

    accessibility: Series[float] = Field(ge=0)
