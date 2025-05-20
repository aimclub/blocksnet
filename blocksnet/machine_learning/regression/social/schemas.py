import shapely
import pandas as pd
from pandera import Field
from pandera.typing import Series
from loguru import logger
from ....utils.validation import DfSchema

class TechnicalIndicatorsSchema(DfSchema):
    footprint_area: Series[float] = Field(ge=0)
    build_floor_area: Series[float] = Field(ge=0)
    living_area: Series[float] = Field(ge=0)
    non_living_area: Series[float] = Field(ge=0)
    population: Series[float] = Field(ge=0)
    site_area: Series[float] = Field(ge=0)
    residential: Series[float] = Field(ge=0)
    recreation: Series[float] = Field(ge=0)
    industrial: Series[float] = Field(ge=0)
    special: Series[float] = Field(ge=0)