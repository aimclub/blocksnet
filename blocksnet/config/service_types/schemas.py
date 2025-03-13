import pandas as pd
from pandera import Field
from pandera.typing import Series, Index
from ...utils.validation import DfSchema


class ServiceTypesSchema(DfSchema):
    idx: Index[str] = Field(str_matches=r"^[a-z]+([_-][a-z]+)*$")
    name_ru: Series[str] = Field(nullable=True)
    demand: Series[int] = Field(ge=0)
    accessibility: Series[int] = Field(ge=0)


class UnitsSchema(DfSchema):
    service_type: Series[str]
    capacity: Series[int]
    site_area: Series[float] = Field(ge=0)
    build_floor_area: Series[float] = Field(ge=0)

    @classmethod
    def _preprocess(cls, df: pd.DataFrame):
        if "parking_area" in df:
            df["site_area"] += df["parking_area"]
        return df
