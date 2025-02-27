import pandas as pd
from pandera import Field
from pandera.typing import Series
from ....utils.validation import DfSchema


class BlocksSchema(DfSchema):

    site_area: Series[float] = Field(ge=0)
    footprint_area: Series[float] = Field(ge=0)
    build_floor_area: Series[float] = Field(ge=0)
    living_area: Series[float] = Field(ge=0)
    business_area: Series[float] = Field(ge=0)
