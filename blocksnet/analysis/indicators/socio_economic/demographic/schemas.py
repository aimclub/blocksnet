from pandera import Field
from pandera.typing import Series
from blocksnet.utils.validation import DfSchema


class BlocksSchema(DfSchema):
    population: Series[int] = Field(ge=0)
    site_area: Series[float] = Field(ge=0)
