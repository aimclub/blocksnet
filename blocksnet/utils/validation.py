from loguru import logger
import pandera as pa
import pandas as pd
import geopandas as gpd
from shapely.geometry.base import BaseGeometry
from pandera.typing import Index
from pandera.typing.geopandas import GeoSeries

DEFAULT_CRS = 4326


class DfSchema(pa.DataFrameModel):
    idx: Index[int] = pa.Field(unique=True)

    class Config:
        strict = "filter"
        add_missing_columns = True
        coerce = True

    @classmethod
    def _check_instance(cls, df):
        if not isinstance(df, gpd.GeoDataFrame):
            raise ValueError("An instance of DataFrame must be provided.")

    @classmethod
    def validate(cls, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        cls._check_instance(df)
        # Проверка мультииндекса и колонок
        if df.index.nlevels > 1:
            raise ValueError("Index must not be multi-leveled.")
        if df.columns.nlevels > 1:
            raise ValueError("Columns must not be multi-leveled.")
        # Вызов стандартной валидации
        return super().validate(df, **kwargs)

    @classmethod
    def _columns(cls) -> list:
        return list(cls.to_schema().columns.keys())

    @classmethod
    def create_empty(cls) -> pd.DataFrame:
        return pd.DataFrame([], columns=cls._columns())

    @pa.dataframe_parser
    @classmethod
    def _enforce_column_order(cls, df: pd.DataFrame):
        return df[cls._columns()]


class GdfSchema(DfSchema):
    geometry: GeoSeries

    @classmethod
    def _geometry_types(cls) -> set[type[BaseGeometry]]:
        raise NotImplementedError

    @classmethod
    def create_empty(cls, crs=DEFAULT_CRS) -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame([], columns=cls._columns(), crs=crs)

    @classmethod
    def _check_instance(cls, df):
        if not isinstance(df, gpd.GeoDataFrame):
            raise ValueError("An instance of GeoDataFrame must be provided.")

    @classmethod
    def _warn_crs(cls, df):
        current_crs = df.crs
        if not current_crs.is_projected:
            recommended_crs = df.estimate_utm_crs()
            logger.warning(
                f"Current CRS {current_crs.to_epsg()} is not projected. It might cause problems when carrying out spatial operations. Recommended: EPSG:{recommended_crs.to_epsg()}."
            )

    @classmethod
    def validate(cls, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        cls._check_instance(df)
        cls._warn_crs(df)
        # Вызов стандартной валидации
        return super().validate(df, **kwargs)

    @pa.check("geometry")
    @classmethod
    def _check_geometry(cls, series) -> pd.Series:
        return series.map(lambda x: any(isinstance(x, geom_type) for geom_type in cls._geometry_types()))


def validate_accessibility_matrix(accessibility_matrix: pd.DataFrame, blocks_df: pd.DataFrame | None = None):
    if not all(accessibility_matrix.index == accessibility_matrix.columns):
        raise ValueError("Matrix index and columns must match")
    if blocks_df is not None:
        if not blocks_df.index.isin(accessibility_matrix.index).all():
            raise ValueError("Block index must be in matrix index")
