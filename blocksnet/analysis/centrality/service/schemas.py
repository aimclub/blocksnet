import pandas as pd
from pandera import Field
from pandera.typing import Series
from ....common.validation import DfSchema


class BlocksSchema(DfSchema):

    site_area: Series[float] = Field(ge=0)
