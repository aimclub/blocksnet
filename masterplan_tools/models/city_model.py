"""
The aim of this module is to create one window to get any required data for other methods.
All data is gathered once and then reused during calculations.
"""

import geopandas as gpd
import networkx as nx
import pandas as pd

from masterplan_tools.method.blocks.blocks_cutter import BlocksCutter
from masterplan_tools.preprocessing.data_getter import DataGetter


class CityModel:  # pylint: disable=too-many-instance-attributes,too-few-public-methods
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

    def __init__(  # pylint: disable=too-many-arguments
        self,
        buildings: gpd.GeoDataFrame | None = None,
        services: dict[gpd.GeoDataFrame] = ...,
        roads_geometry: gpd.GeoDataFrame | None = None,
        water_geometry: gpd.GeoDataFrame | None = None,
        railways_geometry: gpd.GeoDataFrame | None = None,
        nature_geometry_boundaries: gpd.GeoDataFrame | None = None,
        city_geometry: gpd.GeoDataFrame | None = None,
        accessibility_matrix: gpd.GeoDataFrame = ...,
        transport_graph: nx.Graph | None = None,
        greenings: gpd.GeoDataFrame | None = None,
        parkings: gpd.GeoDataFrame | None = None,
        city_blocks: gpd.GeoDataFrame = ...,
    ) -> None:
        """Initialize CityModel

        Args:
            buildings (GeoDataFrame | None, optional): city buildings geometry. Defaults to None.
            services (dict[GeoDataFrame], optional): city services geodataframes dictionary where keys are service
            names and values - their geometry. Defaults to empty dictionary.
            roads_geometry (GeoDataFrame | None, optional): roads GeoDataFrame for blocks generation. Defaults to None.
            water_geometry (GeoDataFrame | None, optional): water GeoDataFrame for blocks generation. Defaults to None.
            railways_geometry (GeoDataFrame | None, optional): railways GeoDataFrame for blocks generation.
            Defaults to None.
            nature_geometry_boundaries (GeoDataFrame | None, optional): nature GeoDataFrame. Defaults to None.
            city_geometry (GeoDataFrame | None, optional): full city geometry. Defaults to None.
            accessibility_matrix (GeoDataFrame, optional): accesibility matrix GeoDataFrame. Defaults to empty
            GeoDataFrame.
            transport_graph (nx.Graph | None, optional): transport graph for provision calculations. Defaults to None.
            greenings (GeoDataFrame | None, optional): green zones areas GeoDataFrame. Defaults to None.
            parkings (GeoDataFrame | None, optional): parking areas GeoDataFrame. Defaults to None.
            city_blocks (GeoDataFrame, optional): city blocks GeoDataFrame (if set, no generation will be performed).
            Defaults to empty GeoDataFrame.
        """
        # TODO: add notes about needed columns for DataFrames/GeoDataFrames/Graphs
        # TODO: Maybe it is more logical to pass city_geometry as shapely.geometry.base.BaseGeometry, not GeoDataFrame?
        if services is ...:
            services = {}
        if accessibility_matrix is ...:
            accessibility_matrix = gpd.GeoDataFrame()
        if city_blocks is ...:
            city_blocks = gpd.GeoDataFrame()
        self.buildings: gpd.GeoDataFrame = buildings
        self.services_gdfs: dict[gpd.GeoDataFrame] = services
        self.water_geometry: gpd.GeoDataFrame = water_geometry
        self.roads_geometry: gpd.GeoDataFrame = roads_geometry
        self.railways_geometry: gpd.GeoDataFrame = railways_geometry
        self.nature_geometry_boundaries: gpd.GeoDataFrame = nature_geometry_boundaries
        """GeoDataFrame of the nature in the city"""
        self.city_geometry: gpd.GeoDataFrame = city_geometry
        """geometry of the city on specified admin level"""
        self.accessibility_matrix = accessibility_matrix
        """
        if the user have pre-caluclated accessibility_matrix, else the matrix will be calculated
        (!) Imortant note: it takes about 40GB RAM to calculate the matris on the intermodal or walk graph
        for the big city like Saint Petersburg
        """
        self.transport_graph: nx.Graph | None = transport_graph
        """
        if there's no specified accessibility matrix, the graph is needed to calculate one.
        For example, the graph could be the drive, bike or walk graph from the OSM
        or the intermodal graph from CityGeoTools
        """
        self.greenings: gpd.GeoDataFrame | None = greenings
        self.parkings: gpd.GeoDataFrame | None = parkings
        self.city_blocks: gpd.GeoDataFrame = city_blocks
        self.blocks_aggregated_info: pd.DataFrame | None = None
        """aggregated info by blocks is needed for further balancing"""
        self.updated_block_info: dict | None = None
        self.services_graph: nx.Graph | None = None
        """updated block is the id of the modified block"""

        self.collect_data()

    def collect_data(self) -> None:
        """
        This method calls DataGetter and BlocksCutter to collect all required data
        to get city blocks and service graphs.
        """

        # Run modelling blocks if they are not provided
        if self.city_blocks.shape[0] == 0:
            self.city_blocks = BlocksCutter(self).cut_blocks()

        # Run modelling accessibility matrix between blocks if it is not provided
        if self.accessibility_matrix.shape[0] == 0:
            self.accessibility_matrix = DataGetter().get_accessibility_matrix(
                blocks=self.city_blocks, graph=self.transport_graph
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
                services_graph=self.services_graph,
            )
