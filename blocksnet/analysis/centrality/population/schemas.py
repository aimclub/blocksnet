import pandas as pd
from pandera import Field, dataframe_check
from pandera.typing import Series
from ....common.validation import DfSchema


class BlocksSchema(DfSchema):

    population: Series[int] = Field(ge=0)
