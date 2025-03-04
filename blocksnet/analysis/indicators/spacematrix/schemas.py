import pandas as pd
from pandera import Field, dataframe_check
from pandera.typing import Series
from ....common.validation import DfSchema


class BlocksSchema(DfSchema):

    site_area: Series[float] = Field(ge=0)
    footprint_area: Series[float] = Field(ge=0)
    build_floor_area: Series[float] = Field(ge=0)
    living_area: Series[float] = Field(ge=0)
    business_area: Series[float] = Field(ge=0)

    # @dataframe_check
    # @classmethod
    # def _validate_bfa_and_fa(cls, df: pd.DataFrame) -> pd.DataFrame:
    #     bfa_ge_fa = all(df.build_floor_area >= df.footprint_area)
    #     if not bfa_ge_fa:
    #         raise ValueError("build_floor_area must be greater than or equal to footprint_area.")
    #     return bfa_ge_fa

    # @dataframe_check
    # @classmethod
    # def _validate_la_and_ba(cls, df: pd.DataFrame) -> pd.DataFrame:
    #     la_and_ba = all(df.living_area + df.business_area <= df.build_floor_area)
    #     if not la_and_ba:
    #         raise ValueError("living_area + business_area must be less than or equal to build_floor_area.")
    #     return la_and_ba
