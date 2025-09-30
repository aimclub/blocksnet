import shapely
import pandas as pd
from pandera import Field
from pandera.typing import Series
from blocksnet.utils.validation import GdfSchema


class ServicesSchema(GdfSchema):
    """ServicesSchema class.

    """
    capacity: Series[float] = Field(nullable=True)

    @classmethod
    def _geometry_types(cls):
        """Geometry types.

        """
        return {shapely.geometry.base.BaseGeometry}
