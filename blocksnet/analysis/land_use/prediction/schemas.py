import pandas as pd
from pandera import Field, check
from pandera.typing import Series
from blocksnet.utils.validation import GdfSchema
from shapely import Polygon, Point


class BlocksInputSchema(GdfSchema):
    land_use: Series[str]
    city: Series[str]
    land_use_code: Series[int] = Field(ge=0)
    city_center: Series[object]

    @classmethod
    def _geometry_types(cls):
        return [Polygon]

    @check("city_center")
    @classmethod
    def _check_city_center_type(cls, s: pd.Series) -> pd.Series:
        return s.map(lambda x: isinstance(x, Point))

