"""
The aim of this module is to create one window to get any required data for other methods.
All data is gathered once and then reused during calculations.
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
        city_name: str,
        city_crs: int,
        city_admin_level: int,
        from_device: bool,
        service_types: list = None,
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
        self.services_gdfs = {}
        self.water_geometry: Optional[gpd.GeoDataFrame] = None
        self.roads_geometry: Optional[gpd.GeoDataFrame] = None
        self.railways_geometry: Optional[gpd.GeoDataFrame] = None
        self.nature_geometry_boundaries: Optional[gpd.GeoDataFrame] = None
        """GeoDataFrame of the nature in the city"""
        self.city_geometry: Optional[gpd.GeoDataFrame] = None
        """geometry of the city on specified admin level"""
        self.city_id: int = city_id
        """city id is specified to be used id db queries"""
        self.service_types = service_types
        """service type must be the same as on the OSM"""
        self.accessibility_matrix = accessibility_matrix
        """
        if the user have pre-caluclated accessibility_matrix, else the matrix will be calculated
        (!) Imortant note: it takes about 40GB RAM to calculate the matris on the intermodal or walk graph
        for the big city like Saint Petersburg
        """
        self.engine = engine
        """engine is used to connect to local db. Else the data will be gathered from OSM"""
        self.graph = graph
        """
        if there's no specified accessibility matrix, the graph is needed to calculate one.
        For example, the graph could be the drive, bike or walk graph from the OSM
        or the intermodal graph from CityGeoTools
        """
        self.transport_graph_type = transport_graph_type
        """transport graph type is the description of the data in the graph"""
        self.blocks_aggregated_info = None
        """aggregated info by blocks is needed for further balancing"""
        self.city_blocks = None
        self.services_graphs = {}
        self.from_device = from_device
        """this argument specifies if the data uploaded from file (output data) or from other source e.g. OSM or db"""
        self.greenings = None
        self.parkings = None
        self.updated_block_info = None
        """updated block is the id of the modified block"""

    def collect_data(self):
        """
        This method calls DataGetter and BlocksCutter to collect all required data
        to get city blocks and service graphs.
        """

        if self.from_device:
            self.city_blocks = gpd.read_parquet("../masterplanning/masterplan_tools/output_data/blocks.parquet")
        else:
            self.city_geometry = DataGetter().get_city_geometry(self.city_name, self.city_admin_level)
            self.roads_geometry = DataGetter().get_roads_geometry(self.city_geometry, roads_buffer=self.ROADS_WIDTH)
            self.railways_geometry = DataGetter().get_railways_geometry(self.city_name, self.RAILWAYS_WIDTH)
            self.nature_geometry_boundaries = DataGetter().get_nature_geometry(
                self.city_name, nature_buffer=self.NATURE_WIDTH, park_cutoff_area=self.PARK_CUTOFF_AREA
            )
            self.water_geometry = DataGetter().get_water_geometry(
                city_name=self.city_name, water_buffer=self.WATER_WIDTH
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

        self.buildings = DataGetter().get_buildings(
            engine=self.engine, city_id=self.city_id, city_crs=self.city_crs, from_device=self.from_device
        )
        self.greenings = DataGetter().get_greenings(engine=self.engine, city_id=self.city_id, city_crs=self.city_crs)
        self.parkings = DataGetter().get_parkings(engine=self.engine, city_id=self.city_id, city_crs=self.city_crs)

        self.blocks_aggregated_info = DataGetter().aggregate_blocks_info(
            blocks=self.city_blocks, buildings=self.buildings, parkings=self.parkings, greenings=self.greenings
        )

        for service_type in self.service_types:
            self.services_gdfs[service_type] = DataGetter().get_service(
                engine=self.engine, city_crs=self.city_crs, city_id=self.city_id, service_type=service_type
            )

        for service_type in self.service_types:
            self.services_graphs[service_type] = DataGetter().prepare_graph(
                blocks=self.city_blocks,
                city_crs=self.city_crs,
                service_type=service_type,
                buildings=self.buildings,
                service_gdf=self.services_gdfs[service_type],
                updated_block_info=None,
                accessibility_matrix=self.accessibility_matrix,
            )
