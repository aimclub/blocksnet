from abc import ABC
from typing import Generic, TypeVar

import geopandas as gpd
from pydantic import BaseModel, ConfigDict
from shapely.geometry.base import BaseGeometry


T = TypeVar("T")


class BaseRow(BaseModel, ABC):
    """Provides an abstract for data validation in GeoDataFrame.
    Generics must be inherited from this base class.

    The inherited class also can be configured to provide default column values to avoid None and NaN"""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    geometry: BaseGeometry
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
        """_summary_

        Parameters
        ----------
        data : _type_
            _description_
        """
        generic_class = self.generic
        assert issubclass(generic_class, BaseRow), "Generic should be inherited from BaseRow"
        # if data is not a GeoDataFrame, we create it ourselves
        if not isinstance(data, gpd.GeoDataFrame):
            data = gpd.GeoDataFrame(data, *args, **kwargs)
        # next we create list of dicts from BaseRow inherited class
        rows: list[dict] = [generic_class(index=i, **data.loc[i].to_dict()).__dict__ for i in data.index]
        super().__init__(rows)
        # and finally return index to where it belongs
        if "index" in self.columns:
            self.index = self["index"]
            self.drop(columns=["index"], inplace=True)
        index_name = data.index.name
        self.index.name = index_name
        self.set_geometry("geometry", inplace=True)
        # and also set crs
        self.crs = kwargs["crs"] if "crs" in kwargs else data.crs
