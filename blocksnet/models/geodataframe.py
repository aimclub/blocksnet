import geopandas as gpd
from typing import Generic, TypeVar
from abc import ABC
from pydantic import BaseModel, Field, ConfigDict
from shapely import Geometry

T = TypeVar("T")


class BaseRow(BaseModel, ABC):
    """Provides an abstract for data validation in GeoDataFrame.
    Generics must be inherited from this base class.

    The inherited class also can be configured to provide default column values to avoid None and NaN"""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    geometry: Geometry
    index: int
    """Index can be override but should not be set by default"""


class GeoDataFrame(gpd.GeoDataFrame, BaseModel, Generic[T]):
    """Basically a geopandas GeoDataFrame, but with Generic[T].
    Provide a BaseRow inherited class to automatically validate data on init."""

    @property
    def generic(self):
        # pydantic is only needed to access generic class
        return self.__pydantic_generic_metadata__["args"][0]

    def __init__(self, data, *args, **kwargs):
        generic_class = self.generic
        assert issubclass(generic_class, BaseRow), "Generic should be inherited from BaseRow"
        # if data is not a GeoDataFrame, we create it ourselves
        if not isinstance(data, gpd.GeoDataFrame):
            data = gpd.GeoDataFrame(data, *args, **kwargs)
        # next we create list of dicts from BaseRow inherited class
        rows: list[dict] = []
        for i in data.index:
            dict = data.loc[i].to_dict()
            rows.append(generic_class(index=i, **dict).__dict__)
        super().__init__(rows)
        # and finally return index to where it belong
        index_name = data.index.name
        self.index = self["index"]
        self.index.name = index_name
        self.drop(columns=["index"], inplace=True)
        # and also set crs if possible
        self.crs = kwargs["crs"] if "crs" in kwargs else data.crs
