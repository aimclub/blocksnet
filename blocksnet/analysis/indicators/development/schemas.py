import pandas as pd
from pandera import Field, dataframe_check
from pandera.typing import Series
from ....common.validation import DfSchema


class BlocksSchema(DfSchema):

    site_area: Series[float] = Field(ge=0)
    fsi: Series[float] = Field(ge=0)
    gsi: Series[float] = Field(ge=0, le=1)
    mxi: Series[float] = Field(ge=0, le=1, default=0)

    @dataframe_check
    @classmethod
    def _validate_fsi_and_gsi(cls, df: pd.DataFrame) -> bool:
        fsi_ge_gsi = all(df.fsi >= df.gsi)
        if not fsi_ge_gsi:
            raise ValueError("FSI must be greater than or equal to GSI.")
        return fsi_ge_gsi
