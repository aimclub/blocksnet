from pandera.typing import Series
from pandera import Field
from blocksnet.utils.validation import DfSchema


class BlocksAreaSchema(DfSchema):

    site_area: Series[float] = Field(ge=0)


class BlocksServicesSchema(DfSchema):

    count: Series[int] = Field(ge=0)
