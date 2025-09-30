import shapely
import pandas as pd
from pandera import Field, parser
from pandera.typing import Series
from blocksnet.utils.validation import GdfSchema, DfSchema
from .common import BlockCategory


class BlocksSchema(GdfSchema):
    @classmethod
    """BlocksSchema class.

    """
    def _geometry_types(cls):
        """Geometry types.

        """
        return {shapely.Polygon}


# class BlocksFeaturesSchema(DfSchema):

#     area : Series[float] = Field(ge=0)
#     length : Series[float] = Field(ge=0)
#     centerline_length : Series[float] = Field(ge=0)
#     corners_count : Series[int] = Field(ge=0)


class BlocksCategoriesSchema(DfSchema):

    """BlocksCategoriesSchema class.

    """
    category: Series

    @parser("category")
    @classmethod
    def _parse_land_use(cls, category: pd.Series) -> pd.Series:
        """Parse land use.

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
