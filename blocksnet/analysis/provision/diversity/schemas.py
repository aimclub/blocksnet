from pandera.typing import Series
from pandera import Field
from ....common.validation import DfSchema


class BlocksServicesSchema(DfSchema):

    count: Series[int] = Field(ge=0, default=0)
