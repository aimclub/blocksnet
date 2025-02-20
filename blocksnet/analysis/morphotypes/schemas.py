from pandera.typing import Series
from pandera import Field
from ...utils.validation import DfSchema


class BlocksSchema(DfSchema):

    l: Series[float] = Field(ge=0, default=0)
    fsi: Series[float] = Field(ge=0, default=0)
    mxi: Series[float] = Field(ge=0, default=0)
