"""
The aim of this module is to create one window to get any required data for other methods.
All data is gathered once and then reused during calculations.
"""

import geopandas as gpd
import networkx as nx
import pandas as pd
import geopandas as gpd
from typing import Literal
from pydantic import BaseModel, Field, InstanceOf, field_validator

from .geojson import PolygonGeoJSON, PointGeoJSON

# from masterplan_tools.method.blocks.blocks_cutter import BlocksCutter
# from masterplan_tools.preprocessing.data_getter import DataGetter
class AccessibilityMatrix(BaseModel):

    df: InstanceOf[pd.DataFrame]

    @field_validator("df", mode="before")
    def validate_df(value):
        assert len(value.columns) == len(value.index), "Size must be NxN"
        assert all(value.columns.unique() == value.index.unique()), "Columns and rows are not equal"
        return value.copy()


class CityBlockFeature(BaseModel):
    landuse: Literal["buildings", "selected_area", "no_dev_area"]
    block_id: int
    current_population: float = Field(ge=0)
    floors: float = Field(ge=0)
    current_living_area: float = Field(ge=0)
    current_green_capacity: float = Field(ge=0)
    current_green_area: float = Field(ge=0)
    current_parking_capacity: float = Field(ge=0)
    current_industrial_area: float = Field(ge=0)
    area: float = Field(ge=0)


class ServicesFeature(BaseModel):
    capacity: int = Field(ge=0)


class CityModel(BaseModel):  # pylint: disable=too-many-instance-attributes,too-few-public-methods
    """
    City model gathers all data in one class so it could be accessed directly in one place
    """

    blocks: PolygonGeoJSON[CityBlockFeature]
    # accessibility_matrix : AccessibilityMatrix
    services: dict[str, PointGeoJSON[ServicesFeature]]

    @field_validator("blocks", mode="before")
    def validate_blocks(value):
        if isinstance(value, gpd.GeoDataFrame):
            return PolygonGeoJSON[CityBlockFeature].from_gdf(value)
        return value

    # @field_validator("accessibility_matrix", mode="before")
    # def validate_matrix(value):
    #     if isinstance(value, pd.DataFrame):
    #         return AccessibilityMatrix(df=value)
    #     return value

    @field_validator("services", mode="before")
    def validate_services(value):
        dict = value.copy()
        for service_type in dict:
            if isinstance(dict[service_type], gpd.GeoDataFrame):
                dict[service_type] = PointGeoJSON[ServicesFeature].from_gdf(value[service_type])
        return dict

    # def collect_data(self) -> None:
    #     """
    #     This method calls DataGetter and BlocksCutter to collect all required data
    #     to get city blocks and service graphs.
    #     """

    #     # Create graphs between living blocks and specified services
    #     self.services_graph = nx.Graph()
    #     for service_type in self.services_gdfs.keys():
    #         self.services_graph = DataGetter().prepare_graph(
    #             blocks=self.city_blocks,
    #             service_type=service_type,
    #             buildings=self.buildings,
    #             service_gdf=self.services_gdfs[service_type],
    #             updated_block_info=None,
    #             accessibility_matrix=self.accessibility_matrix,
    #             services_graph=self.services_graph,
    #         )
