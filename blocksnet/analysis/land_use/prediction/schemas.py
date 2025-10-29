import pandas as pd
from pandera import Field
from pandera.typing import Series
from shapely import MultiPolygon, Polygon
from blocksnet.enums import LandUseCategory
from blocksnet.utils.validation import GdfSchema, LandUseSchema


class BlocksInputSchema(GdfSchema):
    category: Series = Field(nullable=True)

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
                if isinstance(c, str):
                    s = c.strip()
                    # try value (exact), then case-insensitive by value, then by name
                    try:
                        return LandUseCategory(s)
                    except Exception:
                        pass
                    try:
                        return next(v for v in LandUseCategory if v.value.lower() == s.lower())
                    except Exception:
                        pass
                    try:
                        return LandUseCategory[s.upper()]
                    except Exception:
                        pass
                return None

            df["category"] = df["category"].map(parse_category)

        elif "land_use" in df.columns:
            lu_df = LandUseSchema(df)

            def to_category(lu):
                if lu is None:
                    return None
                return LandUseCategory.from_land_use(lu)

            df["category"] = lu_df["land_use"].map(to_category)

        return df

