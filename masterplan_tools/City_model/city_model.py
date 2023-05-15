"""
TODO: add docstring
"""

import geopandas as gpd  # pylint: disable=import-error
import pandas as pd
from typing import Optional
from masterplan_tools.Blocks_getter.blocks_getter import BlocksCutter
from masterplan_tools.Data_getter.data_getter import DataGetter


class CityModel:
    """
    TODO: add docstring
    """

    GLOBAL_CRS = 4326
    """globally used crs."""
    ROADS_WIDTH = RAILWAYS_WIDTH = NATURE_WIDTH = 3
    """road geometry buffer in meters. So road geometries won't be thin as a line."""
    WATER_WIDTH = 1
    """water geometry buffer in meters. So water geometries in some cases won't be thin as a line."""
    GEOMETRY_CUTOFF_RATIO = 0.15
    """polygon's perimeter to area ratio. Objects with bigger ration will be dropped."""
    GEOMETRY_CUTOFF_AREA = 1_400
    """in meters. Objects with smaller area will be dropped."""
    PARK_CUTOFF_AREA = 10_000
    """in meters. Objects with smaller area will be dropped."""

    def __init__(
        self,
        city_name,
        city_crs,
        city_admin_level,
        service_names: list = None,
        city_id: int = None,
        accessibility_matrix=pd.DataFrame(),
        engine=None,
        graph=None,
        transport_graph_type="intermodal",
    ) -> None:
        self.city_name: str = city_name
        self.city_crs: int = city_crs
        self.city_admin_level: int = city_admin_level
        self.buildings = None
        self.services = {}
        self.water_geometry: Optional[gpd.GeoDataFrame] = None
        self.roads_geometry: Optional[gpd.GeoDataFrame] = None
        self.railways_geometry: Optional[gpd.GeoDataFrame] = None
        self.nature_geometry_boundaries: Optional[gpd.GeoDataFrame] = None
        self.city_geometry: Optional[gpd.GeoDataFrame] = None
        self.city_id: int = city_id
        self.service_names = service_names
        self.accessibility_matrix = accessibility_matrix
        self.engine = engine
        self.graph = graph
        self.transport_graph_type = transport_graph_type
        self.blocks_aggregated_info = None
        self.city_blocks = None
        self.services_graphs = {}

    def collect_data(self):
        """
        TODO: add docstring
        """

        self.city_geometry = DataGetter().get_city_geometry(self.city_name, self.city_admin_level)
        self.roads_geometry = DataGetter().get_roads_geometry(self.city_geometry, roads_buffer=self.ROADS_WIDTH)
        self.railways_geometry = DataGetter().get_railways_geometry(self.city_name, self.RAILWAYS_WIDTH)
        self.nature_geometry_boundaries = DataGetter().get_nature_geometry(
            self.city_name, nature_buffer=self.NATURE_WIDTH, park_cutoff_area=self.PARK_CUTOFF_AREA
        )
        self.water_geometry = DataGetter().get_water_geometry(city_name=self.city_name, water_buffer=self.WATER_WIDTH)

        for service in self.service_names:
            self.services[service] = DataGetter().get_service(
                engine=self.engine, city_crs=self.city_crs, city_id=self.city_id, service_type=service
            )

        self.city_blocks = BlocksCutter(self).get_blocks()

        if self.accessibility_matrix.shape[0] == 0:
            self.accessibility_matrix = DataGetter().get_accessibility_matrix(
                city_crs=self.city_crs, blocks=self.city_geometry, G=self.graph, option=self.transport_graph_type
            )

        self.water_geometry = None
        self.roads_geometry = None
        self.railways_geometry = None
        self.nature_geometry_boundaries = None
        self.city_geometry = None

        self.blocks_aggregated_info = DataGetter().aggregate_blocks_info(
            blocks=self.city_blocks, engine=self.engine, city_crs=self.city_crs, city_id=self.city_id
        )

        for service in self.service_names:
            self.services_graphs[service] = DataGetter().prepare_graph(self, blocks=self.city_blocks, engine=self.engine, city_id= self.city_id, 
            city_crs =self.city_crs, from_device=False, service_type=service, updated_block_info = None, 
            accessibility_matrix = self.accessibility_matrix
            )

