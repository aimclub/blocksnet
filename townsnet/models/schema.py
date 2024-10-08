import pandera as pa
import geopandas as gpd
from shapely import Point, MultiPoint, Polygon, MultiPolygon, LineString, MultiLineString
from pandera.typing import Series, Index
from pandera.typing.geopandas import GeoSeries


class BaseSchema(pa.DataFrameModel):
    idx: Index[int] = pa.Field(unique=True)
    geometry: GeoSeries
    _geom_types = [Point, MultiPoint, Polygon, MultiPolygon, LineString, MultiLineString]

    class Config:
        strict = "filter"
        add_missing_columns = True

    @classmethod
    def to_gdf(cls):
        columns = cls.to_schema().columns.keys()
        return gpd.GeoDataFrame(data=[], columns=columns, crs=4326)

    @pa.check("geometry")
    @classmethod
    def check_geometry(cls, series):
        return series.map(lambda x: any([isinstance(x, geom_type) for geom_type in cls._geom_types]))