from pandera import Field
from pandera.typing import Series
from ...utils.validation import DfSchema


class BlocksSchema(DfSchema):
    site_area: Series[float] = Field(ge=0)
    population: Series[int] = Field(ge=0)


class ServicesSchema(DfSchema):
    capacity: Series[int] = Field(ge=0)
