from __future__ import annotations
import pickle
import networkx as nx
import geopandas as gpd
import pandas as pd
from shapely import Polygon, LineString
from matplotlib import pyplot as plt
from functools import singledispatchmethod
from pydantic import BaseModel, Field, InstanceOf
from .service_type import ServiceType

SERVICE_TYPES = {
    "kindergartens": {"demand": 61, "accessibility": 10, "buffer": 15},
    "schools": {"demand": 120, "accessibility": 15},
    "recreational_areas": {"demand": 6000, "accessibility": 15},
    "hospitals": {"demand": 9, "accessibility": 60},
    "pharmacies": {"demand": 50, "accessibility": 10},
    "policlinics": {"demand": 27, "accessibility": 15},
}


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
    capacities: dict[ServiceType, int] = {}
    """Service type aggregated capacity value"""
    city: InstanceOf[City]
    """City instance that contains block"""

    def to_dict(self) -> dict[str, int]:
        return {"id": self.id, "geometry": self.geometry, "population": self.population}

    def __getitem__(self, service_type_name: str) -> int:
        service_type = self.city[service_type_name]
        result = {"capacity": 0, "demand": service_type.calculate_in_need(self.population)}
        if service_type in self.capacities:
            result["capacity"] = self.capacities[service_type]
        return result

    def update_capacity(self, service_type: ServiceType, capacity):
        self.capacities[service_type] = capacity

    @property
    def is_living(self):
        return self.population > 0

    @classmethod
    def from_gdf(cls, gdf: gpd.GeoDataFrame, city: City) -> "list[Block]":
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
            .apply(lambda x: cls(**x.to_dict(), city=city), axis=1)
            .to_list()
        )

    def __hash__(self):
        return hash(self.id)


class City:
    epsg: int
    graph: nx.DiGraph
    service_types: list[ServiceType] = []

    def plot(self, max_weight: int = 5) -> None:
        """Plot city model blocks and relations"""
        blocks = self.get_blocks_gdf()
        ax = blocks.plot(alpha=1, color="#ddd")
        edges = []
        for u, v, data in self.graph.edges(data=True):
            a = u.geometry.representative_point()
            b = v.geometry.representative_point()
            if data["weight"] <= max_weight:
                edges.append({"geometry": LineString([a, b]), "weight": data["weight"]})
        gpd.GeoDataFrame(edges).set_crs(self.epsg).plot(ax=ax, alpha=0.2, column="weight", cmap="cool", legend=True)
        ax.set_axis_off()

    @property
    def blocks(self) -> list[Block]:
        return list(self.graph.nodes)

    def get_blocks_gdf(self) -> gpd.GeoDataFrame:
        data: list[dict] = []
        for block in self.blocks:
            data.append(block.to_dict())
        gdf = gpd.GeoDataFrame(data).set_index("id").set_crs(epsg=self.epsg)
        return gdf

    def update_layer(self, service_type: ServiceType | str, gdf: gpd.GeoDataFrame):
        """Updates blocks with relevant information about city services"""
        if not isinstance(service_type, ServiceType):
            service_type = self[service_type]
        sjoin = gdf.to_crs(epsg=self.epsg).sjoin(self.get_blocks_gdf())
        sjoin = sjoin.groupby("index_right").agg({"capacity": "sum"})
        for block_id in sjoin.index:
            self[block_id].update_capacity(service_type, sjoin.loc[block_id, "capacity"])

    def add_service_type(self, name: str, accessibility: int = None, demand: int = None):
        if name in self:
            raise KeyError(f"The service type with this name already exists: {name}")
        else:
            self.service_types.append(ServiceType(name=name, accessibility=accessibility, demand=demand))

    @singledispatchmethod
    def __getitem__(self, arg):
        raise NotImplementedError(f"Can't access object with such argument type {type(arg)}")

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
            raise KeyError(f"Can't find service type with such name: {service_type_name}")
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

    @staticmethod
    def from_pickle(file_path: str):
        """Load city model from a .pickle file"""
        state = None
        with open(file_path, "rb") as f:
            state = pickle.load(f)
        return state

    def to_pickle(self, file_path: str):
        """Save city model to a .pickle file"""
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    def __init__(
        self, matrix: pd.DataFrame, blocks_gdf: gpd.GeoDataFrame, services: dict[str, gpd.GeoDataFrame] = {}
    ) -> None:
        self.epsg = blocks_gdf.crs.to_epsg()
        blocks = Block.from_gdf(blocks_gdf, self)
        graph = nx.DiGraph()
        for i in matrix.index:
            graph.add_edge(blocks[i], blocks[i], weight=matrix.loc[i, i])
            for j in matrix.columns.drop(i):
                graph.add_edge(blocks[i], blocks[j], weight=matrix.loc[i, j])
        self.graph = graph
        self.service_types = list(map(lambda name: ServiceType(name=name, **SERVICE_TYPES[name]), SERVICE_TYPES))
        for service_type, gdf in services.items():
            self.update_layer(service_type, gdf)
