from pandera.typing import Series
from pandera import Field
from blocksnet.utils.validation import DfSchema


class BlocksServicesSchema(DfSchema):

    """BlocksServicesSchema class.

    """
    count: Series[int] = Field(ge=0, default=0)
