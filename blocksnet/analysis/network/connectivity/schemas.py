from pandera.typing import Series
from pandera import Field
from blocksnet.utils.validation import DfSchema


class BlocksAccessibilitySchema(DfSchema):
    """BlocksAccessibilitySchema class.

    """
    accessibility: Series[float] = Field(ge=0)
