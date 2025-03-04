import shapely
import pandas as pd
from pandera import dataframe_parser
from ...common.validation import GdfSchema


class BlocksSchema(GdfSchema):
    @classmethod
    def _geometry_types(cls):
        return {shapely.Polygon}
