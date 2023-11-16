from __future__ import annotations
import pickle
import networkx as nx
import geopandas as gpd
import pandas as pd
from shapely import Polygon, LineString, Point
from matplotlib import pyplot as plt
from functools import singledispatchmethod
from pydantic import BaseModel, Field, InstanceOf
from .service_type import ServiceType
from .geodataframe import GeoDataFrame, BaseRow

SERVICE_TYPES = {
    "kindergartens": {"demand": 61, "accessibility": 10},
    "schools": {"demand": 120, "accessibility": 15},
    "recreational_areas": {"demand": 6000, "accessibility": 15},
    "hospitals": {"demand": 9, "accessibility": 60},
    "pharmacies": {"demand": 50, "accessibility": 10},
    "policlinics": {"demand": 27, "accessibility": 15},
}


class BuildingRow(BaseRow):
    geometry: Point
    population: int = Field(ge=0, default=0)
    floors: float = Field(ge=0)
    area: float = Field(ge=0)
    living_area: float = Field(ge=0)
    is_living: bool


class ServiceRow(BaseRow):
    geometry: Point
    capacity: int = Field(ge=0)


class Block(BaseModel):
    """Class presenting city block"""

    id: int
    """Unique block identifier across the ```city```"""
    geometry: InstanceOf[Polygon] = Field()
    """Block geometry presented as shapely ```Polygon```"""
    landuse: str = None
    """Current city block landuse"""
    buildings: InstanceOf[gpd.GeoDataFrame] = None
    """Buildings ```GeoDataFrame```"""
    services: InstanceOf[dict[ServiceType, gpd.GeoDataFrame]] = {}
    """Services ```GeoDataFrames```s for different ```ServiceType```s"""
    city: InstanceOf[City]
    """```City``` instance that contains the block"""

    def to_dict(self, simplify=True) -> dict[str, int]:
        dict = {"id": self.id, "geometry": self.geometry}
        if not simplify:
            for service_type in self.services:
                dict[service_type.name] = self[service_type.name]["capacity"]
            dict["population"] = self.population
            dict["is_living"] = self.is_living
        return dict

    def __contains__(self, service_type_name: str) -> bool:
        """Returns True if service type is contained inside the block"""
        service_type = self.city[service_type_name]
        return service_type in self.services

    def __getitem__(self, service_type_name: str) -> dict[str, int]:
        """Get service type capacity and demand of the block"""
        service_type = self.city[service_type_name]
        result = {"capacity": 0, "demand": service_type.calculate_in_need(self.population)}
        if service_type in self.services:
            result["capacity"] = self.services[service_type]["capacity"].sum()
        return result

    def update_buildings(self, gdf: GeoDataFrame[BuildingRow] = None):
        """Update buildings GeoDataFrame of the block"""
        if gdf is None:
            self.buildings = None
        else:
            self.buildings = gpd.GeoDataFrame(gdf)

    def update_services(self, service_type: ServiceType, gdf: GeoDataFrame[ServiceRow] = None):
        """Update services GeoDataFrame of the block"""
        if gdf is None:
            del self.services[service_type]
        else:
            self.services[service_type] = gpd.GeoDataFrame(gdf)

    @property
    def is_living(self) -> bool:
        if self.buildings is not None:
            return self.buildings.is_living.any()
        else:
            return False

    @property
    def population(self) -> int:
        """Return sum population of the city block"""
        if self.buildings is not None:
            return self.buildings.population.sum()
        else:
            return 0

    @classmethod
    def from_gdf(cls, gdf: gpd.GeoDataFrame, city: City) -> "list[Block]":
        """Generate blocks list from ```GeoDataFrame```"""
        return gdf.apply(lambda x: cls(id=x.name, **x.to_dict(), city=city), axis=1).to_list()

    def __hash__(self):
        """Make block hashable, so it can be used as a node in ```nx.Graph```"""
        return hash(self.id)


class City:
    epsg: int
    graph: nx.DiGraph
    service_types: list[ServiceType]

    def __init__(self, blocks_gdf: gpd.GeoDataFrame, adjacency_matrix: pd.DataFrame) -> None:
        self.epsg = blocks_gdf.crs.to_epsg()
        blocks = Block.from_gdf(blocks_gdf, self)
        graph = nx.DiGraph()
        for i in adjacency_matrix.index:
            for j in adjacency_matrix.columns:
                graph.add_edge(blocks[i], blocks[j], weight=adjacency_matrix.loc[i, j])
        self.graph = graph
        self.service_types = list(map(lambda name: ServiceType(name=name, **SERVICE_TYPES[name]), SERVICE_TYPES))

    def plot(self, max_weight: int = 5) -> None:
        """Plot city model blocks and relations"""
        blocks = self.get_blocks_gdf()
        ax = blocks.plot(alpha=1, color="#ddd")
        edges = gpd.GeoDataFrame(self.graph.edges(data=True), columns=["u", "v", "data"])
        edges["weight"] = edges["data"].apply(lambda x: x["weight"])
        edges = edges.loc[edges["weight"] <= max_weight]
        edges = edges.sort_values(by="weight", ascending=False)
        edges["geometry"] = edges.apply(
            lambda x: LineString([x.u.geometry.representative_point(), x.v.geometry.representative_point()]), axis=1
        )
        edges = edges.set_geometry("geometry").set_crs(self.epsg).drop(columns=["u", "v", "data"])
        edges.plot(ax=ax, alpha=0.2, column="weight", cmap="cool", legend=True)
        ax.set_axis_off()

    @property
    def blocks(self) -> list[Block]:
        return list(self.graph.nodes)

    def get_blocks_gdf(self, simplify=True) -> gpd.GeoDataFrame:
        data: list[dict] = []
        for block in self.blocks:
            data.append(block.to_dict(simplify))
        gdf = gpd.GeoDataFrame(data).set_index("id").set_crs(epsg=self.epsg)
        return gdf

    def update_buildings(self, gdf: gpd.GeoDataFrame):
        """Update buildings in blocks"""
        assert gdf.crs.to_epsg() == self.epsg, "Buildings GeoDataFrame CRS should match city EPSG"
        # reset buildings of blocks
        for block in self.blocks:
            block.update_buildings()
        # spatial join blocks and buildings and updated related blocks info
        sjoin = gdf.sjoin(self.get_blocks_gdf())
        groups = sjoin.groupby("index_right")
        for block_id, buildings_gdf in groups:
            self[block_id].update_buildings(GeoDataFrame[BuildingRow](buildings_gdf))

    def update_services(self, service_type: ServiceType | str, gdf: gpd.GeoDataFrame):
        """Update services in blocks of certain service_type"""
        assert gdf.crs.to_epsg() == self.epsg, "Services GeoDataFrame CRS should match city EPSG"
        if not isinstance(service_type, ServiceType):
            service_type = self[service_type]
        # reset services of blocks
        for block in filter(lambda b: service_type.name in b, self.blocks):
            block.update_services(service_type)
        # spatial join blocks and services and update related blocks info
        sjoin = gdf.sjoin(self.get_blocks_gdf())
        groups = sjoin.groupby("index_right")
        for block_id, services_gdf in groups:
            self[block_id].update_services(service_type, GeoDataFrame[ServiceRow](services_gdf))

    def add_service_type(self, name: str, accessibility: int = None, demand: int = None):
        if name in self:
            raise KeyError(f"The service type with this name already exists: {name}")
        else:
            self.service_types.append(ServiceType(name=name, accessibility=accessibility, demand=demand))

    def get_distance(self, block_a: int | Block, block_b: int | Block):
        """Returns distance (in min) between two blocks in directed graph"""
        if not isinstance(block_a, Block):
            block_a = self[block_a]
        if not isinstance(block_b, Block):
            block_b = self[block_b]
        return self.graph[block_a][block_b]["weight"]

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
