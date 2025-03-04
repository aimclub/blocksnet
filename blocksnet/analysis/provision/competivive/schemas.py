from pandera.typing import Series
from pandera import Field
from ....common.validation import DfSchema


class BlocksSchema(DfSchema):

    population: Series[int] = Field(ge=0)
    capacity: Series[int] = Field(ge=0)
