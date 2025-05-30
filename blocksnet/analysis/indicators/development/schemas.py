import pandas as pd
from pandera import Field, dataframe_check
from pandera.typing import Series

from ....utils.validation import DfSchema


class BlocksSchema(DfSchema):

    site_area: Series[float] = Field(ge=0)
    fsi: Series[float] = Field(ge=0)
    gsi: Series[float] = Field(ge=0, le=1)
    mxi: Series[float] = Field(ge=0, le=1, default=0)
