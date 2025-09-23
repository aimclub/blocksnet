from pandera import Field
from pandera.typing import Series
from blocksnet.utils.validation import DfSchema


class CountSchema(DfSchema):
    count: Series[int] = Field(ge=0)
