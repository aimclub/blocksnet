from loguru import logger
from typing import cast
import pyproj
import pandera as pa
import pandas as pd
import geopandas as gpd
from shapely.geometry.base import BaseGeometry
from pandera.typing.geopandas import GeoSeries
from .df_schema import DfSchema

DEFAULT_CRS = 4326


class GdfSchema(DfSchema):
    geometry: GeoSeries

    def __new__(cls, *args, **kwargs) -> gpd.GeoDataFrame:
        return cast(gpd.GeoDataFrame, cls.validate(*args, **kwargs))

    @classmethod
    def create_empty(cls, crs: pyproj.CRS | int | None = DEFAULT_CRS) -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame([], columns=cls.columns_(), crs=crs)

    @classmethod
    def _check_instance(cls, gdf):
        if not isinstance(gdf, gpd.GeoDataFrame):
            raise ValueError("An instance of GeoDataFrame must be provided")

    @pa.dataframe_parser
    @classmethod
    def _warn_crs(cls, gdf):
        current_crs = gdf.crs
        if current_crs is None:
            logger.warning("Current CRS is None. Further operations might be invalid")
        elif not current_crs.is_projected:
            recommended_crs = gdf.estimate_utm_crs()
            logger.warning(
                f"Current CRS {current_crs.to_epsg()} is not projected. It might cause problems when carrying out spatial operations. Recommended: EPSG:{recommended_crs.to_epsg()}"
            )
        return gdf

    @classmethod
    def _geometry_types(cls) -> set[type[BaseGeometry]]:
        raise NotImplementedError

    @pa.check("geometry")
    @classmethod
    def _check_geometry(cls, series) -> pd.Series:
        return series.map(lambda x: any(isinstance(x, geom_type) for geom_type in cls._geometry_types()))
