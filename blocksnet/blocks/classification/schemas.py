from enum import Enum
import shapely
import pandas as pd
from pandera import Field, parser, check
from pandera.typing import Series
from blocksnet.utils.validation import GdfSchema, DfSchema


class BlockCategory(Enum):
    INVALID = "invalid"
    LARGE = "large"
    NORMAL = "normal"


class BlocksGeometriesSchema(GdfSchema):

    site_area: Series[float] = Field(ge=0)
    site_length: Series[float] = Field(ge=0)

    @classmethod
    def _before_validate(cls, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "geometry" in df.columns:
            if not "site_area" in df.columns:
                df["site_area"] = df["geometry"].area
            if not "site_length" in df.columns:
                df["site_length"] = df["geometry"].length
        return df

    @classmethod
    def _geometry_types(cls):
        return [shapely.Polygon]


class BlocksCategoriesSchema(DfSchema):

    category: Series

    @parser("category")
    @classmethod
    def _parse_category(cls, category: pd.Series) -> pd.Series:
        return category.apply(lambda c: BlockCategory(c.lower()) if isinstance(c, str) else c)

    @check("category")
    @classmethod
    def _check_category(cls, category: pd.Series) -> pd.Series:
        return category.apply(lambda c: isinstance(c, BlockCategory))
