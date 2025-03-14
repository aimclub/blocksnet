import pandas as pd
from pandera import Field, parser, check
from pandera.typing import Series
from ....enums import LandUse
from ....utils.validation import LandUseSchema


class BlocksSchema(LandUseSchema):
    site_area: Series[float] = Field(ge=0)
