import shapely
from loguru import logger
from pandera import Field
from pandera.typing import Series
from ....utils.validation import GdfSchema


class BuildingsSchema(GdfSchema):
    is_living: Series[bool]
    number_of_floors: Series[float] = Field(ge=0, nullable=True)
    footprint_area: Series[float] = Field(ge=0, nullable=True)
    build_floor_area: Series[float] = Field(ge=0, nullable=True)
    living_area: Series[float] = Field(ge=0, nullable=True)
    non_living_area: Series[float] = Field(ge=0, nullable=True)
    population: Series[float] = Field(ge=0, nullable=True)

    @classmethod
    def _before_validate(cls, df):
        for column in [c for c in cls.columns_() if c != "is_living"]:
            if not column in df:
                logger.warning(f"Column {column} not found and will be initialized as None")
        return df

    @classmethod
    def _geometry_types(cls):
        return {shapely.geometry.base.BaseGeometry}
