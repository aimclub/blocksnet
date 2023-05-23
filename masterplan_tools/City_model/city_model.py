"""
The aim of this module is to create one window to get any required data for other methods.
All data is gathered once and then reused during calculations.
"""

from typing import Dict
import networkx as nx
import pandas as pd
import geopandas as gpd
from masterplan_tools.Blocks_getter.blocks_getter import BlocksCutter
from masterplan_tools.Data_getter.data_getter import DataGetter

 
class CityModel:
    """
    City model gathers all data in one class so it could be accessed directly in one place
    """

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

    def __init__(self, **kwargs) -> None:
        self.buildings: gpd.GeoDataFrame = kwargs.get("buildings", None)
        self.services_gdfs: Dict[gpd.GeoDataFrame] = kwargs.get("services", {})
        self.water_geometry: gpd.GeoDataFrame = kwargs.get("water_geometry", None)
        self.roads_geometry: gpd.GeoDataFrame = kwargs.get("roads_geometry", None)
        self.railways_geometry: gpd.GeoDataFrame = kwargs.get("railways_geometry", None)
        self.nature_geometry_boundaries: gpd.GeoDataFrame = kwargs.get("nature_geometry_boundaries", None)
        """GeoDataFrame of the nature in the city"""
        self.city_geometry: gpd.GeoDataFrame = kwargs.get("city_geometry", None)
        """geometry of the city on specified admin level"""
        self.accessibility_matrix = kwargs.get("accessibility_matrix", pd.DataFrame())
        """
            if the user have pre-caluclated accessibility_matrix, else the matrix will be calculated
            (!) Imortant note: it takes about 40GB RAM to calculate the matris on the intermodal or walk graph
            for the big city like Saint Petersburg
            """
        self.transport_graph: nx.Graph = kwargs.get("transport_graph", None)
        """
            if there's no specified accessibility matrix, the graph is needed to calculate one.
            For example, the graph could be the drive, bike or walk graph from the OSM
            or the intermodal graph from CityGeoTools
            """
        self.greenings: gpd.GeoDataFrame = kwargs.get("greenings", None)
        self.parkings: gpd.GeoDataFrame = kwargs.get("parkings", None)
        self.city_blocks: gpd.GeoDataFrame = kwargs.get("city_blocks", pd.DataFrame())
        self.blocks_aggregated_info: pd.DataFrame = None
        """aggregated info by blocks is needed for further balancing"""
        self.services_graphs: Dict[gpd.GeoDataFrame] = {}
        self.updated_block_info: dict = None
        self.services_graph: nx.Graph() = None
        """updated block is the id of the modified block"""

        self.collect_data()

    def collect_data(self):
        """
        This method calls DataGetter and BlocksCutter to collect all required data
        to get city blocks and service graphs.
        """

        # Run modelling blocks if they are not provided
        if self.city_blocks.shape[0] == 0:
            self.city_blocks = BlocksCutter(self).get_blocks()

        # Run modelling accessibility matrix between blocks if it is not provided
        if self.accessibility_matrix.shape[0] == 0:
            self.accessibility_matrix = DataGetter().get_accessibility_matrix(
                blocks=self.city_blocks, G=self.transport_graph
            )

        # Get aggregated information about blocks
        # This information will be used during modelling new parameters for blocks
        self.blocks_aggregated_info = DataGetter().aggregate_blocks_info(
            blocks=self.city_blocks,
            buildings=self.buildings,
            parkings=self.parkings,
            greenings=self.greenings,
        )


        # Create graphs between living blocks and specified services
        self.services_graph = nx.Graph()
        for service_type in self.services_gdfs.keys():
            self.services_graph = DataGetter().prepare_graph(
                blocks=self.city_blocks,
                service_type=service_type,
                buildings=self.buildings,
                service_gdf=self.services_gdfs[service_type],
                updated_block_info=None,
                accessibility_matrix=self.accessibility_matrix,
                services_graph = self.services_graph
            )
