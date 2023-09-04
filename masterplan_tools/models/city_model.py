"""
The aim of this module is to create one window to get any required data for other methods.
All data is gathered once and then reused during calculations.
"""

import geopandas as gpd
import networkx as nx
import pandas as pd
import geopandas as gpd
from typing import Literal, Optional
from pydantic import BaseModel, Field, InstanceOf, field_validator
from shapely import LineString
import matplotlib.pyplot as plt

from .geojson import PolygonGeoJSON, PointGeoJSON

# from masterplan_tools.preprocessing.utils import Utils

# from masterplan_tools.method.blocks.blocks_cutter import BlocksCutter
class AccessibilityMatrix(BaseModel):
    """
    Accessibility matrix between city blocks
    """

    df: InstanceOf[pd.DataFrame]
    """
    Accessibility matrix DataFrame
    """

    @field_validator("df", mode="before")
    def validate_df(value):
        assert len(value.columns) == len(value.index), "Size must be NxN"
        assert all(value.columns.unique() == value.index.unique()), "Columns and rows are not equal"
        return value.copy()


class CityBlockFeature(BaseModel):
    """
    Aggregated city block feature properties
    """

    landuse: Literal["buildings", "selected_area", "no_dev_area"]
    """Landuse label, containing one of the next values:
    1. 'no_dev_area' -- according to th no_debelopment gdf and cutoff without any buildings or specified / selected landuse types;
    2. 'selected_area' -- according to the landuse gdf. We separate theese polygons since they have specified landuse types;
    3. 'buildings' -- there are polygons that have buildings landuse type.
    """
    block_id: int
    """Unique city block identifier"""
    is_living: bool
    """Is block living"""
    current_population: float = Field(ge=0)
    """Total population of the block"""
    floors: float = Field(ge=0)
    """Median storeys count of the buildings inside the block"""
    current_living_area: float = Field(ge=0)
    """Total living area of the block (in square meters)"""
    current_green_capacity: float = Field(ge=0)
    """Total greenings capacity (in units)"""
    current_green_area: float = Field(ge=0)
    """Total greenings area (in square meters)"""
    current_parking_capacity: float = Field(ge=0)
    """Total parkings capacity (in units)"""
    current_industrial_area: float = Field(ge=0)
    """Total industrial area of the block (in square meters)"""
    area: float = Field(ge=0)
    """Total area of the block (in square meters)"""


class ServicesFeature(BaseModel):
    """
    Service feature properties
    """

    capacity: int = Field(ge=0)
    """Total service object capacity"""


class CityModel(BaseModel):  # pylint: disable=too-many-instance-attributes,too-few-public-methods
    """
    City representation as an information model
    """

    blocks: PolygonGeoJSON[CityBlockFeature]
    """Aggregated city blocks"""
    accessibility_matrix: AccessibilityMatrix
    """Accessibility matrix between city blocks"""
    services: dict[str, PointGeoJSON[ServicesFeature]]
    """Services geometries of the city"""
    services_graph: InstanceOf[nx.Graph] = Field(exclude=True, default=None)
    """nx.Graph of the city, containing provision assessment capacities"""

    def get_service_types(self) -> list[str]:
        return list(self.services.keys())

    @field_validator("blocks", mode="before")
    def validate_blocks(value):
        if isinstance(value, gpd.GeoDataFrame):
            return PolygonGeoJSON[CityBlockFeature].from_gdf(value)
        return value

    @field_validator("accessibility_matrix", mode="before")
    def validate_matrix(value):
        if isinstance(value, pd.DataFrame):
            return AccessibilityMatrix(df=value)
        return value

    @field_validator("services", mode="before")
    def validate_services(value):
        dict = value.copy()
        for service_type in dict:
            if isinstance(dict[service_type], gpd.GeoDataFrame):
                dict[service_type] = PointGeoJSON[ServicesFeature].from_gdf(value[service_type])
        return dict

    def visualize(self, max_distance=7) -> None:
        """Method for city model visualization"""
        blocks = self.blocks.to_gdf()
        centroids = blocks.copy()
        centroids["geometry"] = centroids["geometry"].centroid
        edges = []
        for u, v, a in self.services_graph.edges(data=True):
            if a["weight"] < max_distance and a["weight"] > 0:
                edges.append(
                    {
                        "distance": a["weight"],
                        "geometry": LineString([centroids.loc[u, "geometry"], centroids.loc[v, "geometry"]]),
                    }
                )
        edges = gpd.GeoDataFrame(edges).sort_values(ascending=False, by="distance")
        fig, ax = plt.subplots(figsize=(15, 15))
        blocks.plot(ax=ax, alpha=0.5, color="#ddd")
        edges.plot(ax=ax, alpha=0.1, column="distance", cmap="summer")
        plt.show()

    def prepare_graph(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        service_type: str,
        services_graph: nx.Graph,
        updated_block_info: dict = None,
    ):
        """
        This function prepares a graph for calculating the provision of a specified service in a city.

        Args:
            blocks (gpd.GeoDataFrame): A GeoDataFrame containing information about the blocks in the city.
            service_type (str, optional): The type of service to calculate the provision for. Defaults to None.
            service_gdf (gpd.GeoDataFrame, optional): A GeoDataFrame containing information about blocks with the
            specified service in the city. Defaults to None.
            accessibility_matrix (np.ndarray, optional): An accessibility matrix for the city. Defaults to None.
            buildings (gpd.GeoDataFrame, optional): A GeoDataFrame containing information about buildings in the city.
            Defaults to None.
            updated_block_info (gpd.GeoDataFrame, optional): A GeoDataFrame containing updated information
            about blocks in the city. Defaults to None.

        Returns:
            nx.Graph: A networkx graph representing the city's road network with additional data for calculating
            the provision of the specified service.
        """

        blocks = self.blocks.to_gdf()
        service = self.services[service_type].to_gdf()
        accessibility_matrix = self.accessibility_matrix.df.copy()

        blocks.rename(columns={"current_population": "population_balanced", "block_id": "id"}, inplace=True)
        blocks["is_living"] = blocks["population_balanced"].apply(lambda x: x > 0)

        living_blocks = blocks.loc[:, ["id", "geometry"]].sort_values(by="id").reset_index(drop=True)

        service_gdf = (
            gpd.sjoin(blocks, service, predicate="intersects")
            .groupby("id")
            .agg(
                {
                    "capacity": "sum",
                }
            )
        )

        if updated_block_info:
            for updated_block in updated_block_info.values():
                if updated_block["block_id"] not in service_gdf.index:
                    service_gdf.loc[updated_block["block_id"]] = 0
                if service_type == "recreational_areas":
                    service_gdf.loc[updated_block["block_id"], "capacity"] += updated_block.get("G_max_capacity", 0)
                else:
                    service_gdf.loc[updated_block["block_id"], "capacity"] += updated_block.get(
                        f"{service_type}_capacity", 0
                    )

            blocks.loc[updated_block["block_id"], "population_balanced"] = updated_block["population"]

        blocks_geom_dict = blocks[["id", "population_balanced", "is_living"]].set_index("id").to_dict()
        service_blocks_dict = service_gdf.to_dict()["capacity"]

        blocks_list = accessibility_matrix.loc[
            accessibility_matrix.index.isin(service_gdf.index.astype("Int64")),
            accessibility_matrix.columns.isin(living_blocks["id"]),
        ]

        # TODO: is tqdm really necessary?
        for idx in list(blocks_list.index):
            blocks_list_tmp = blocks_list[blocks_list.index == idx]
            blocks_list.columns = blocks_list.columns.astype(int)
            blocks_list_tmp_dict = blocks_list_tmp.transpose().to_dict()[idx]

            for key in blocks_list_tmp_dict.keys():
                if key != idx:
                    services_graph.add_edge(idx, key, weight=round(blocks_list_tmp_dict[key], 1))

                else:
                    services_graph.add_node(idx)

                services_graph.nodes[key]["population"] = blocks_geom_dict["population_balanced"][int(key)]
                services_graph.nodes[key]["is_living"] = blocks_geom_dict["is_living"][int(key)]

                if key != idx:
                    try:
                        if services_graph.nodes[key][f"is_{service_type}_service"] != 1:
                            services_graph.nodes[key]["id"] = key
                            services_graph.nodes[key][f"is_{service_type}_service"] = 0
                            services_graph.nodes[key][f"provision_{service_type}"] = 0
                            services_graph.nodes[key][f"id_{service_type}"] = 0
                            services_graph.nodes[key][f"{service_type}_capacity"] = 0

                    except KeyError:
                        services_graph.nodes[key]["id"] = key
                        services_graph.nodes[key][f"is_{service_type}_service"] = 0
                        services_graph.nodes[key][f"provision_{service_type}"] = 0
                        services_graph.nodes[key][f"id_{service_type}"] = 0
                        services_graph.nodes[key][f"{service_type}_capacity"] = 0

                else:
                    services_graph.nodes[key]["id"] = key
                    services_graph.nodes[key][f"is_{service_type}_service"] = 1
                    services_graph.nodes[key][f"{service_type}_capacity"] = service_blocks_dict[key]
                    services_graph.nodes[key][f"provision_{service_type}"] = 0
                    services_graph.nodes[key][f"id_{service_type}"] = 0

                if services_graph.nodes[key]["is_living"]:
                    services_graph.nodes[key][f"population_prov_{service_type}"] = 0
                    services_graph.nodes[key][f"population_unprov_{service_type}"] = blocks_geom_dict[
                        "population_balanced"
                    ][int(key)]

        return services_graph

    def model_post_init(self, __context) -> None:
        values = self.dict()
        services = values["services"]
        services_graph = nx.Graph()
        for service_type in services.keys():
            services_graph = self.prepare_graph(
                service_type=service_type,
                services_graph=services_graph,
            )
        self.services_graph = services_graph
