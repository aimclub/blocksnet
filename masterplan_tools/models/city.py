import networkx as nx
import geopandas as gpd
import pandas as pd
from matplotlib import pyplot as plt
from shapely import Polygon, LineString
from pydantic import BaseModel, Field, InstanceOf
from functools import singledispatchmethod
import math

SERVICE_TYPES = {
    "kindergartens": {"demand": 61, "accessibility": 10},
    "schools": {"demand": 120, "accessibility": 15},
    "recreational_areas": {"demand": 6000, "accessibility": 15},
    "hospitals": {"demand": 9, "accessibility": 60},
    "pharmacies": {"demand": 50, "accessibility": 10},
    "policlinics": {"demand": 27, "accessibility": 15},
}


class ServiceType(BaseModel):
    """Represents service type entity, such as schools and its parameters overall"""

    name: str
    accessibility: int = Field(gt=0)
    demand: int = Field(gt=0)

    def calculate_in_need(self, population: int) -> int:
        return math.ceil(population / 1000 * self.demand)

    def __hash__(self):
        return hash(self.name)


class AggregatedService(BaseModel):
    """Represents service type information of a block"""

    service_type: InstanceOf[ServiceType]
    capacity: int = Field(ge=0)


class Block(BaseModel):
    id: int
    geometry: InstanceOf[Polygon]
    population: int = Field(ge=0)
    floors: float = Field(ge=0)
    area: float = Field(ge=0)
    living_area: float = Field(ge=0)
    green_area: float = Field(ge=0)
    industrial_area: float = Field(ge=0)
    green_capacity: int = Field(ge=0)
    parking_capacity: int = Field(ge=0)
    aggregated_services: list[AggregatedService] = []

    def __getitem__(self, service_type_name: str) -> int:
        items = list(filter(lambda x: x.service_type.name == service_type_name, self.aggregated_services))
        if len(items) == 0:
            return 0
        return {"capacity": items[0].capacity, "demand": items[0].service_type.calculate_in_need(self.population)}

    def __contains__(self, service_type_name):
        return service_type_name in [x.service_type.name for x in self.aggregated_services]

    def add_services(self, service_type: ServiceType, gdf: gpd.GeoDataFrame):
        capacity = gdf[gdf["geometry"].apply(self.geometry.contains)].copy()["capacity"].sum()
        self.aggregated_services.append(AggregatedService(service_type=service_type, capacity=capacity))

    @property
    def is_living(self):
        return self.population > 0

    @classmethod
    def from_gdf(cls, gdf: gpd.GeoDataFrame) -> "list[Block]":
        return (
            gdf.rename(
                columns={
                    "block_id": "id",
                    "current_population": "population",
                    "current_living_area": "living_area",
                    "current_green_capacity": "green_capacity",
                    "current_green_area": "green_area",
                    "current_parking_capacity": "parking_capacity",
                    "current_industrial_area": "industrial_area",
                },
                inplace=False,
            )
            .apply(lambda x: cls(**x.to_dict()), axis=1)
            .to_list()
        )

    def __hash__(self):
        return hash(self.id)


class City:
    epsg: int
    graph: nx.DiGraph
    service_types: list[ServiceType] = []

    def plot(self) -> None:
        """Plot city model blocks and relations"""
        # blocks = self.blocks.to_gdf()
        # centroids = blocks.copy()
        # centroids["geometry"] = centroids["geometry"].centroid
        # edges = []
        # for u, v, a in self.services_graph.edges(data=True):
        #     if a["weight"] <  and a["weight"] > 0:
        #         edges.append(
        #             {
        #                 "distance": a["weight"],
        #                 "geometry": LineString([centroids.loc[u, "geometry"], centroids.loc[v, "geometry"]]),
        #             }
        #         )
        # edges = gpd.GeoDataFrame(edges).sort_values(ascending=False, by="distance")
        # fig, ax = plt.subplots(figsize=(15, 15))
        # blocks.plot(ax=ax, alpha=0.5, color="#ddd")
        # edges.plot(ax=ax, alpha=0.1, column="distance", cmap="summer")
        # plt.show()

    @property
    def blocks(self) -> list[Block]:
        return list(self.graph.nodes)

    def get_blocks_gdf(self) -> gpd.GeoDataFrame:
        data: list[dict] = []
        for block in self.blocks:
            data.append({"id": block.id, "geometry": block.geometry})
        gdf = gpd.GeoDataFrame(data).set_index("id").set_crs(epsg=self.epsg)
        return gdf

    def update_service_type_layer(self, service_type: ServiceType, gdf: gpd.GeoDataFrame):
        crs_gdf = gdf.to_crs(epsg=self.epsg)
        for block in self.blocks:
            block.add_services(service_type=service_type, gdf=crs_gdf)

    def add_service_type(self, name: str, accessibility: int = None, demand: int = None):
        if name in self:
            raise KeyError(f"The service type with this name already exists: {name}")
        else:
            self.service_types.append(ServiceType(name=name, accessibility=accessibility, demand=demand))

    @singledispatchmethod
    def __getitem__(self, arg):
        raise NotImplementedError(f"Can't access block or service type with such argument type {type(arg)}")

    # Make city_model subscriptable, to access block via ID like city_model[123]
    @__getitem__.register(int)
    def _(self, block_id):
        items = list(filter(lambda x: x.id == block_id, self.blocks))
        if len(items) == 0:
            raise KeyError(f"Can't find block with such id: {block_id}")
        return items[0]

    # Make city_model subscriptable, to access service type via name like city_model['schools']
    @__getitem__.register(str)
    def _(self, service_type_name):
        items = list(filter(lambda x: x.name == service_type_name, self.service_types))
        if len(items) == 0:
            raise KeyError("Can't find service type with such name")
        return items[0]

    @singledispatchmethod
    def __contains__(self, arg):
        raise NotImplementedError(f"Wrong argument type for 'in': {type(arg)}")

    # Make 'in' check available for blocks, to access like 123 in city_model
    @__contains__.register(int)
    def _(self, block_id):
        return block_id in [x.id for x in self.blocks]

    # Make 'in' check available for service types, to access like 'schools' in city_model
    @__contains__.register(str)
    def _(self, service_type_name):
        return service_type_name in [x.name for x in self.service_types]

    def __init__(self, matrix: pd.DataFrame, blocks_gdf: gpd.GeoDataFrame) -> None:
        self.epsg = blocks_gdf.crs.to_epsg()
        blocks = Block.from_gdf(blocks_gdf)
        graph = nx.DiGraph()
        for i in matrix.index:
            graph.add_edge(blocks[i], blocks[i], weight=matrix.loc[i, i])
            for j in matrix.columns.drop(i):
                graph.add_edge(blocks[i], blocks[j], weight=matrix.loc[i, j])
        self.graph = graph
        for name in SERVICE_TYPES:
            self.service_types.append(ServiceType(name=name, **SERVICE_TYPES[name]))
