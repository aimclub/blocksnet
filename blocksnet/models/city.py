from __future__ import annotations

import pickle
from functools import singledispatchmethod

import geopandas as gpd
import networkx as nx
import pandas as pd
from matplotlib import pyplot as plt
from pydantic import BaseModel, Field, InstanceOf, field_validator
from shapely import Point, Polygon
from shapely.geometry.base import BaseGeometry

from ..utils import SERVICE_TYPES
from .geodataframe import BaseRow, GeoDataFrame
from .service_type import ServiceType
from .land_use import LandUse
from blocksnet.models import land_use


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


class LandUseRow(BaseRow):
    geometry: Point
    land_use: LandUse

    @field_validator("geometry", mode="before")
    def validate_geometry(value: BaseGeometry):
        return value.representative_point()

    @field_validator("land_use", mode="before")
    def validate_land_use(value):
        assert isinstance(value, str), "land_use should be str"
        value = value.lower()
        value = value.replace("-", "_")
        return value


class Block(BaseModel):
    """Class presenting city block"""

    id: int
    """Unique block identifier across the ```city```"""
    geometry: InstanceOf[Polygon]
    """Block geometry presented as shapely ```Polygon```"""
    land_use: LandUse | None = None
    """Current city block landuse"""
    buildings: InstanceOf[gpd.GeoDataFrame] = None
    """Buildings ```GeoDataFrame```"""
    services: InstanceOf[dict[ServiceType, gpd.GeoDataFrame]] = {}
    """Services ```GeoDataFrames```s for different ```ServiceType```s"""
    city: InstanceOf[City]
    """```City``` instance that contains the block"""

    @property
    def area(self):
        return self.geometry.area

    @property
    def industrial_area(self):
        if self.buildings is not None:
            return self.buildings.area.sum()
        else:
            return 0

    @property
    def living_area(self):
        if self.buildings is not None:
            return self.buildings.living_area.sum()
        else:
            return 0

    @property
    def land_use_service_types(self) -> list[ServiceType]:
        assert self.land_use != None, "Block land use is unknown (None)"
        service_types = self.city.service_types
        filtered = filter(lambda st: self.land_use in st.land_use, service_types)
        return list(filtered)

    def to_dict(self, simplify=True) -> dict[str, int]:
        dict = {"id": self.id, "geometry": self.geometry}
        if not simplify:
            for service_type in self.services:
                dict[service_type.name] = self[service_type.name]["capacity"]
            dict["population"] = self.population
            dict["is_living"] = self.is_living
            if isinstance(self.land_use, LandUse):
                dict["land_use"] = self.land_use.value
            else:
                dict["land_use"] = None
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
    def from_gdf(cls, gdf: gpd.GeoDataFrame, city: City) -> "dict[int, Block]":
        """Generate blocks dict from ```GeoDataFrame```"""
        dict = {}
        for i in gdf.index:
            dict[i] = cls(id=i, geometry=gdf.loc[i].geometry, city=city)
        return dict

    def __hash__(self):
        """Make block hashable, so it can be used as key in dict etc."""
        return hash(self.id)


class City:
    epsg: int
    adjacency_matrix: pd.DataFrame
    _blocks: dict[int, Block]
    _service_types: dict[str, ServiceType]

    def __init__(self, blocks_gdf: gpd.GeoDataFrame, adjacency_matrix: pd.DataFrame) -> None:
        assert (blocks_gdf.index == adjacency_matrix.index).all(), "Matrix and blocks index don't match"
        assert (blocks_gdf.index == adjacency_matrix.columns).all(), "Matrix columns and blocks index don't match"
        self.epsg = blocks_gdf.crs.to_epsg()
        self._blocks = Block.from_gdf(blocks_gdf, self)
        self.adjacency_matrix = adjacency_matrix.copy()
        self._service_types = {}
        for st in SERVICE_TYPES:
            service_type = ServiceType(**st)
            self._service_types[service_type.name] = service_type

    @property
    def blocks(self) -> list[Block]:
        """Return list of blocks"""
        return [self._blocks[id] for id in self._blocks]

    @property
    def service_types(self) -> list[ServiceType]:
        """Return list of service types"""
        return [self._service_types[name] for name in self._service_types]

    def __str__(self):
        description = ""
        description += f"CRS:          : EPSG:{self.epsg}\n"
        description += f"Blocks count  : {len(self.blocks)}\n"
        description += f"Service types : \n"
        service_types_description = "\n".join([f"    {st}" for st in self.service_types])
        return description + service_types_description

    def plot(self) -> None:
        """Plot city model data"""
        blocks = self.get_blocks_gdf(simplify=False)
        # get gdfs
        no_lu_blocks = blocks.loc[~blocks.land_use.notna()]
        lu_blocks = blocks.loc[blocks.land_use.notna()]
        buildings_gdf = self.get_buildings_gdf()
        services_gdf = self.get_services_gdf()

        # plot
        _, ax = plt.subplots(figsize=(10, 10))
        if len(no_lu_blocks) > 0:
            no_lu_blocks.plot(ax=ax, alpha=1, color="#ddd")
        if len(lu_blocks) > 0:
            lu_blocks.plot(ax=ax, column="land_use", legend=True)
        if len(buildings_gdf) > 0:
            buildings_gdf.plot(ax=ax, markersize=1, color="#bbb")
        if len(services_gdf) > 0:
            services_gdf.plot(
                ax=ax,
                markersize=5,
                column="service_type",
                legend=True,
                legend_kwds={"title": "Service types", "loc": "lower left"},
            )
        ax.set_axis_off()

    def get_service_type_gdf(self, service_type: ServiceType | str):
        if not isinstance(service_type, ServiceType):
            service_type = self[service_type]
        services_blocks = filter(lambda b: service_type in b.services, self.blocks)
        services_gdfs = list(map(lambda b: b.services[service_type].to_crs(4326), services_blocks))
        gdf = gpd.GeoDataFrame(columns=["geometry", "capacity", "service_type"]).set_geometry("geometry").set_crs(4326)
        gdf = pd.concat([gdf, *services_gdfs], ignore_index=True)
        gdf["service_type"] = service_type.name
        return gdf

    def get_buildings_gdf(self) -> gpd.GeoDataFrame:
        buildings_blocks = filter(lambda b: b.buildings is not None, self.blocks)
        buildings_gdfs = list(map(lambda b: b.buildings.to_crs(4326), buildings_blocks))
        gdf = gpd.GeoDataFrame(columns=["geometry"]).set_geometry("geometry").set_crs(4326)
        gdf = pd.concat([gdf, *buildings_gdfs], ignore_index=True).to_crs(self.epsg)
        return gdf

    def get_services_gdf(self) -> gpd.GeoDataFrame:
        gdfs = map(lambda st: self.get_service_type_gdf(st).to_crs(4326), self.service_types)
        return pd.concat(gdfs, axis=0, ignore_index=True).to_crs(self.epsg)

    def get_blocks_gdf(self, simplify=True) -> gpd.GeoDataFrame:
        data: list[dict] = []
        for block in self.blocks:
            data.append(block.to_dict(simplify))
        gdf = gpd.GeoDataFrame(data).set_index("id").set_crs(epsg=self.epsg)
        return gdf

    def update_land_use(self, gdf: GeoDataFrame[LandUseRow]):
        gdf = GeoDataFrame[LandUseRow](gdf)
        blocks_gdf = self.get_blocks_gdf()
        sjoin = blocks_gdf.sjoin(gdf, how="left")
        for i in sjoin.index:
            self[i].land_use = sjoin.loc[i, "land_use"]
        # for block in self.blocks:
        #     block.land_use = gdf.loc[block.id, 'land_use']

    def update_buildings(self, gdf: GeoDataFrame[BuildingRow]):
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

    def update_services(self, service_type: ServiceType | str, gdf: GeoDataFrame[ServiceRow]):
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

    def add_service_type(self, service_type: ServiceType):
        if service_type.name in self:
            raise KeyError(f"The service type with this name already exists: {service_type.name}")
        else:
            self._service_types[service_type.name] = service_type

    def get_distance(self, block_a: int | Block, block_b: int | Block):
        """Returns distance (in min) between two blocks"""
        if isinstance(block_a, Block):
            block_a = block_a.id
        if isinstance(block_b, Block):
            block_b = block_b.id
        return self.adjacency_matrix.loc[block_a, block_b]

    def get_out_edges(self, block: int | Block):
        """Get out edges for certain block"""
        if isinstance(block, Block):
            block = block.id
        return [(self[block], self[block_b], weight) for block_b, weight in self.adjacency_matrix.loc[block].items()]

    def get_in_edges(self, block: int | Block):
        """Get in edges for certain block"""
        if isinstance(block, Block):
            block = block.id
        return [(self[block_b], self[block], weight) for block_b, weight in self.adjacency_matrix.loc[:, block].items()]

    @singledispatchmethod
    def __getitem__(self, arg):
        raise NotImplementedError(f"Can't access object with such argument type {type(arg)}")

    # Make city_model subscriptable, to access block via ID like city_model[123]
    @__getitem__.register(int)
    def _(self, block_id):
        if not block_id in self._blocks:
            raise KeyError(f"Can't find block with such id: {block_id}")
        return self._blocks[block_id]

    # Make city_model subscriptable, to access service type via name like city_model['schools']
    @__getitem__.register(str)
    def _(self, service_type_name):
        if not service_type_name in self._service_types:
            raise KeyError(f"Can't find service type with such name: {service_type_name}")
        return self._service_types[service_type_name]

    @__getitem__.register(tuple)
    def _(self, blocks):
        (block_a_id, block_b_id) = blocks
        block_a = self[block_a_id]
        block_b = self[block_b_id]
        return self.adjacency_matrix.loc[block_a.id, block_b.id]

    @singledispatchmethod
    def __contains__(self, arg):
        raise NotImplementedError(f"Wrong argument type for 'in': {type(arg)}")

    # Make 'in' check available for blocks, to access like 123 in city_model
    @__contains__.register(int)
    def _(self, block_id):
        return block_id in self._blocks

    # Make 'in' check available for service types, to access like 'schools' in city_model
    @__contains__.register(str)
    def _(self, service_type_name):
        return service_type_name in self._service_types

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
