import shapely
import pandas as pd
from pandera import Field
from pandera.typing import Series
from loguru import logger
from ....enums import LandUse
from ....utils.validation import DfSchema, GdfSchema, LandUseSchema


class BlocksSchema(GdfSchema):
    @classmethod
    def _geometry_types(cls):
        return {shapely.Polygon}

class TechnicalIndicatorsSchema(DfSchema):
    residential: Series[float] = Field(ge=0, le=1)
    business: Series[float] = Field(ge=0, le=1)
    recreation: Series[float] = Field(ge=0, le=1)
    industrial: Series[float] = Field(ge=0, le=1)
    transport: Series[float] = Field(ge=0, le=1)
    special: Series[float] = Field(ge=0, le=1)
    agriculture: Series[float] = Field(ge=0, le=1)
