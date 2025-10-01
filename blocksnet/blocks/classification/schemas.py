import shapely
import pandas as pd
from pandera import parser, check
from pandera.typing import Series
from blocksnet.utils.validation import GdfSchema, DfSchema
from blocksnet.enums import BlockCategory


class BlocksSchema(GdfSchema):
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
