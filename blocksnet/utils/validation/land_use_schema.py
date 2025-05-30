import pandas as pd
from pandera import Field, check, parser
from pandera.typing import Series

from ...enums import LandUse
from .df_schema import DfSchema


class LandUseSchema(DfSchema):
    land_use: Series = Field(nullable=True)

    @parser("land_use")
    @classmethod
    def _parse_land_use(cls, land_use: pd.Series) -> pd.Series:
        return land_use.apply(lambda lu: LandUse(lu.lower()) if isinstance(lu, str) else lu)

    @check("land_use")
    @classmethod
    def _check_land_use(cls, land_use: pd.Series) -> pd.Series:
        return land_use.apply(lambda lu: True if lu is None else isinstance(lu, LandUse))
