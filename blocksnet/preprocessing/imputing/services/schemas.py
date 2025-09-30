import shapely
import pandas as pd
from pandera import Field
from pandera.typing import Series
from blocksnet.utils.validation import GdfSchema


class ServicesSchema(GdfSchema):
    """Schema describing service locations used in capacity imputation."""

    capacity: Series[float] = Field(nullable=True)

    @classmethod
    def _geometry_types(cls):
        return {shapely.geometry.base.BaseGeometry}
