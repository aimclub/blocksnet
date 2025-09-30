import pandas as pd
from pandera import Field
from pandera.typing import Series
from loguru import logger
from ....utils.validation import DfSchema


class BlocksSchema(DfSchema):
    """Schema validating block attributes required for density indicators."""

    site_area: Series[float] = Field(ge=0)
    footprint_area: Series[float] = Field(ge=0)
    build_floor_area: Series[float] = Field(ge=0)
    living_area: Series[float] = Field(ge=0)
    non_living_area: Series[float] = Field(ge=0)

    @classmethod
    def _before_validate(cls, df: pd.DataFrame) -> pd.DataFrame:
        if "non_living_area" not in df.columns:
            logger.warning("The non_living_area is not in columns, restoring")
            if "living_area" in df.columns and "build_floor_area" in df.columns:
                df["non_living_area"] = df["build_floor_area"] - df["living_area"]
        return df
