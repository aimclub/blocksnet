from pandera import Field
from pandera.typing import Series
from blocksnet.utils.validation import DfSchema


class BlocksSchema(DfSchema):
    """Schema ensuring demographic indicators have population and area data."""

    population: Series[int] = Field(ge=0)
    site_area: Series[float] = Field(ge=0)
