import pandas as pd
from pandera import Field, check
from pandera.typing import Series
from blocksnet.enums import LandUse, LandUseCategory
from blocksnet.utils.validation import GdfSchema, LandUseSchema
from shapely import Polygon, Point


class BlocksInputSchema(GdfSchema, LandUseSchema):
    city: Series[str]
    city_center: Series[object]
    category: Series[object]

    @classmethod
    def _geometry_types(cls):
        return [Polygon]

    @check("city_center")
    @classmethod
    def _check_city_center_type(cls, s: pd.Series) -> pd.Series:
        return s.map(lambda x: isinstance(x, Point))
    

    @classmethod
    def _before_validate(cls, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # 1. Если есть колонка 'category', парсим её в enum LandUseCategory
        if "category" in df.columns:
            def parse_category(c):
                if isinstance(c, LandUseCategory):
                    return c
                try:
                    return LandUseCategory(c.lower())
                except (ValueError, AttributeError):
                    return None

            df["category"] = df["category"].map(parse_category)

            if df["category"].isnull().any():
                raise ValueError(
                    "Некорректные значения в колонке 'category'. "
                    "Ожидаются: 'urban', 'non_urban', 'industrial' или соответствующие enum-значения."
                )

            return df

        # 2. Если есть колонка 'land_use', парсим и маппим в category
        if "land_use" in df.columns:
            def parse_land_use(val):
                if isinstance(val, LandUse):
                    return val
                if isinstance(val, str):
                    try:
                        return LandUse(val.lower())
                    except ValueError:
                        return None
                return None

            df["land_use"] = df["land_use"].map(parse_land_use)

            def to_category(lu):
                if lu is None:
                    return None
                return LandUseCategory.from_land_use(lu)

            df["category"] = df["land_use"].map(to_category)

            if df["category"].isnull().any():
                raise ValueError(
                    "Некоторые значения 'land_use' не удалось замапить в категорию. "
                    "Проверьте допустимость значений и наличие соответствия в LU_MAPPING."
                )

            return df

        # 3. Ни 'category', ни 'land_use' нет — ошибка
        raise ValueError("В DataFrame должна быть колонка 'category' или 'land_use'.")

    @classmethod
    def _after_validate(cls, df: pd.DataFrame) -> pd.DataFrame:
        df.drop(columns=["land_use"], inplace=True)
        return df
