import pandas as pd
from pandera import Field
from pandera.typing import Series, Index
from ...utils.validation import DfSchema
from ...enums import LandUse

SERVICE_TYPE_NAME_REGEX = r"^[a-z]+([_-][a-z]+)*$"


class ServiceTypesSchema(DfSchema):
    idx: Index[str] = Field(str_matches=SERVICE_TYPE_NAME_REGEX)
    name_ru: Series[str] = Field(nullable=True)
    demand: Series[int] = Field(ge=0)
    accessibility: Series[int] = Field(ge=0)


class UnitsSchema(DfSchema):
    service_type: Series[str] = Field(str_matches=SERVICE_TYPE_NAME_REGEX)
    capacity: Series[int]
    site_area: Series[float] = Field(ge=0)
    build_floor_area: Series[float] = Field(ge=0)

    @classmethod
    def _before_validate(cls, df: pd.DataFrame):
        if "parking_area" in df:
            df["site_area"] += df["parking_area"]
        return df


class LandUseSchema(DfSchema):
    idx: Index[str] = Field(str_matches=SERVICE_TYPE_NAME_REGEX)

    residential: Series[bool] = Field(default=False)
    business: Series[bool] = Field(default=False)
    recreation: Series[bool] = Field(default=False)
    industrial: Series[bool] = Field(default=False)
    transport: Series[bool] = Field(default=False)
    special: Series[bool] = Field(default=False)
    agriculture: Series[bool] = Field(default=False)

    @classmethod
    def _before_validate(cls, df: pd.DataFrame):
        if "land_use" in df.columns:
            for lu_value in [lu.value for lu in list(LandUse)]:
                df[lu_value] = df["land_use"].apply(lambda arr: lu_value in arr)
        return df

    @classmethod
    def _after_validate(cls, df: pd.DataFrame):
        return df[[lu.value for lu in list(LandUse)]].rename(columns=LandUse)
