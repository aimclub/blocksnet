import pandas as pd
import shapely
from pandera import Field
from pandera.typing import Series

from blocksnet.utils.validation import GdfSchema


class ServicesSchema(GdfSchema):
    capacity: Series[float] = Field(nullable=True)

    @classmethod
    def _geometry_types(cls):
        return {shapely.geometry.base.BaseGeometry}
