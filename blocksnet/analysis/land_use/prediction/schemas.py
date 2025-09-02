import pandas as pd
from pandera.typing import Series
from shapely import MultiPolygon, Polygon
from blocksnet.enums import LandUseCategory
from blocksnet.utils.validation import GdfSchema, LandUseSchema


class BlocksInputSchema(GdfSchema):
    category: Series

    @classmethod
    def _geometry_types(cls):
        return [Polygon | MultiPolygon]

    @classmethod
    def _before_validate(cls, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if "category" in df.columns:
            def parse_category(c):
                if isinstance(c, LandUseCategory):
                    return c
                return LandUseCategory(c.lower())

            df["category"] = df["category"].map(parse_category)

        elif "land_use" in df.columns:
            lu_df = LandUseSchema(df)

            def to_category(lu):
                if lu is None:
                    return None
                return LandUseCategory.from_land_use(lu)

            df["category"] = lu_df["land_use"].map(to_category)

        return df

