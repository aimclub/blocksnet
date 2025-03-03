import pandas as pd
import shapely
from pandera import Field, parser, dataframe_check
from pandera.typing import Series
from ...land_use.enum import LandUse
from ...utils.validation import DfSchema, GdfSchema


class BlocksSchema(GdfSchema):
    @classmethod
    def _geometry_types(cls):
        return {shapely.Polygon}


class BlocksLandUseSchema(DfSchema):

    land_use: Series[str] = Field(nullable=True)

    @parser("land_use")
    @classmethod
    def _parse_land_use(cls, series: pd.Series) -> pd.Series:
        return series.apply(lambda s: None if s is None else s.lower())


class BlocksIndicatorsSchema(DfSchema):

    fsi: Series[float] = Field(ge=0, default=0)
    gsi: Series[float] = Field(ge=0, le=1, default=0)
    mxi: Series[float] = Field(ge=0, le=1, default=0)

    @dataframe_check
    @classmethod
    def _validate_fsi_and_gsi(cls, df: pd.DataFrame) -> bool:
        fsi_ge_gsi = all(df.fsi >= df.gsi)
        if not fsi_ge_gsi:
            raise ValueError("FSI must be greater than or equal to GSI.")
        return fsi_ge_gsi
