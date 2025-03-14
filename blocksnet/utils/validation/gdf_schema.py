from loguru import logger
import pandera as pa
import pandas as pd
import geopandas as gpd
from shapely.geometry.base import BaseGeometry
from pandera.typing.geopandas import GeoSeries
from .df_schema import DfSchema

DEFAULT_CRS = 4326


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

    @pa.dataframe_parser
    @classmethod
    def _warn_crs(cls, df):
        current_crs = df.crs
        if not current_crs.is_projected:
            recommended_crs = df.estimate_utm_crs()
            logger.warning(
                f"Current CRS {current_crs.to_epsg()} is not projected. It might cause problems when carrying out spatial operations. Recommended: EPSG:{recommended_crs.to_epsg()}."
            )
        return df

    @pa.check("geometry")
    @classmethod
    def _check_geometry(cls, series) -> pd.Series:
        return series.map(lambda x: any(isinstance(x, geom_type) for geom_type in cls._geometry_types()))
