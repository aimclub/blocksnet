import pandas as pd
import shapely
from pandera import Field
from pandera.typing import Series
from blocksnet.utils.validation import DfSchema, LandUseSchema, GdfSchema
from loguru import logger


class BlocksSchema(GdfSchema):
    """Schema validating block geometries for development imputation."""

    @classmethod
    def _geometry_types(cls):
        return [shapely.Polygon]


class BlocksLandUseSchema(DfSchema):
    """Schema describing land-use composition features for blocks."""

    residential: Series[float] = Field(ge=0, le=1)
    business: Series[float] = Field(ge=0, le=1)
    recreation: Series[float] = Field(ge=0, le=1)
    industrial: Series[float] = Field(ge=0, le=1)
    transport: Series[float] = Field(ge=0, le=1)
    special: Series[float] = Field(ge=0, le=1)
    agriculture: Series[float] = Field(ge=0, le=1)

    @classmethod
    def _before_validate(cls, df):
        if all([col not in df.columns for col in cls.columns_()]):
            logger.warning(f"Not valid format. Trying to one hot from land_use column")
            df = LandUseSchema(df)
            df = df["land_use"].apply(lambda lu: {} if lu is None else lu.to_one_hot()).apply(pd.Series).fillna(0)
        return df


class BlocksIndicatorsSchema(DfSchema):
    """Schema for development indicators predicted by the imputer."""

    build_floor_area: Series[float] = Field(ge=0)
    footprint_area: Series[float] = Field(ge=0)
    living_area: Series[float] = Field(ge=0)
