import shapely
import pandas as pd
from pandera import Field
from pandera.typing import Series
from loguru import logger
from ....enums import LandUse
from ....utils.validation import DfSchema, GdfSchema, LandUseSchema


class BlocksSchema(GdfSchema):
    @classmethod
    """BlocksSchema class.

    """
    def _geometry_types(cls):
        """Geometry types.

        """
        return {shapely.Polygon}


class BlocksGeometriesSchema(DfSchema):
    """BlocksGeometriesSchema class.

    """
    area: Series[float] = Field(ge=0)
    length: Series[float] = Field(ge=0)


class BlocksLandUseSchema(DfSchema):
    """BlocksLandUseSchema class.

    """
    residential: Series[float] = Field(ge=0, le=1)
    business: Series[float] = Field(ge=0, le=1)
    recreation: Series[float] = Field(ge=0, le=1)
    industrial: Series[float] = Field(ge=0, le=1)
    transport: Series[float] = Field(ge=0, le=1)
    special: Series[float] = Field(ge=0, le=1)
    agriculture: Series[float] = Field(ge=0, le=1)

    @classmethod
    def _before_validate(cls, df):
        """Before validate.

        Parameters
        ----------
        df : Any
            Description.

        """
        if all([col not in df.columns for col in cls.columns_()]):
            logger.warning(f"Not valid format. Trying to one hot from land_use column")
            df = LandUseSchema(df)
            df = df["land_use"].apply(lambda lu: {} if lu is None else lu.to_one_hot()).apply(pd.Series).fillna(0)
        return df


class BlocksDensitiesSchema(DfSchema):

    """BlocksDensitiesSchema class.

    """
    fsi: Series[float] = Field(gt=0, default=0)
    gsi: Series[float] = Field(gt=0, le=1, default=0)
    mxi: Series[float] = Field(ge=0, le=1, default=0)
