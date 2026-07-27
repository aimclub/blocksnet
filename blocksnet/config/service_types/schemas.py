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
        # Reject units that occupy neither site area nor build floor area:
        # the optimizer treats them as a degenerate case (sort key divides by
        # ``build_floor_area`` for ``site_area == 0`` units). Validating at
        # load time keeps the run-time invariant intact.
        bad = df[(df.get("site_area", 0) == 0) & (df.get("build_floor_area", 0) == 0)]
        if not bad.empty:
            offenders = bad.assign(_label=bad.index.astype(str))._label.tolist()
            raise ValueError(
                "Service units with both site_area=0 and build_floor_area=0 are not allowed: "
                f"{offenders}"
            )
        # Embedded units (site_area == 0) cannot absorb a parking footprint in
        # the standard ``site_area`` channel; fold ``parking_area`` into the
        # build floor area as a deterministic, capacity-neutral surcharge so
        # the parking demand participates in BFA constraints instead of being
        # silently dropped. For units with a real site footprint, parking
        # extends the on-site footprint.
        if "parking_area" in df.columns:
            embedded_mask = df["site_area"] == 0
            df.loc[embedded_mask, "build_floor_area"] = (
                df.loc[embedded_mask, "build_floor_area"] + df.loc[embedded_mask, "parking_area"]
            )
            df.loc[~embedded_mask, "site_area"] = (
                df.loc[~embedded_mask, "site_area"] + df.loc[~embedded_mask, "parking_area"]
            )
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
