import shapely
import pandas as pd
from pandera import parser, check
from pandera.typing import Series
from blocksnet.utils.validation import GdfSchema, DfSchema
from blocksnet.enums import BlockCategory


class BlocksSchema(GdfSchema):
    @classmethod
    """BlocksSchema class.

    """
    def _geometry_types(cls):
        """Geometry types.

        """
        return [shapely.Polygon]


class BlocksCategoriesSchema(DfSchema):

    """BlocksCategoriesSchema class.

    """
    category: Series

    @parser("category")
    @classmethod
    def _parse_category(cls, category: pd.Series) -> pd.Series:
        """Parse category.

        Parameters
        ----------
        category : pd.Series
            Description.

        Returns
        -------
        pd.Series
            Description.

        """
        return category.apply(lambda c: BlockCategory(c.lower()) if isinstance(c, str) else c)

    @check("category")
    @classmethod
    def _check_category(cls, category: pd.Series) -> pd.Series:
        """Check category.

        Parameters
        ----------
        category : pd.Series
            Description.

        Returns
        -------
        pd.Series
            Description.

        """
        return category.apply(lambda c: isinstance(c, BlockCategory))
