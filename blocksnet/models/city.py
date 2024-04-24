from __future__ import annotations
from abc import ABC
import math

import pickle
from functools import singledispatchmethod
import statistics
import warnings

import geopandas as gpd
import networkx as nx
import pandas as pd
from matplotlib import pyplot as plt
from pydantic import BaseModel, Field, InstanceOf, field_validator, ConfigDict, model_validator
from shapely import Point, Polygon, MultiPolygon, intersection

from ..utils import SERVICE_TYPES
from .geodataframe import BaseRow, GeoDataFrame
from .service_type import ServiceType
from .land_use import LandUse
from blocksnet.models import land_use


class Service(ABC, BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, allow_inf_nan=False)
    service_type: ServiceType
    capacity: int = Field(gt=0)
    area: float = Field(gt=0)
    """Service area in square meters"""

    def to_dict(self) -> dict:
        return {
            "service_type": self.service_type.name,
            "capacity": self.capacity,
            "area": self.area,
        }


class BlockService(Service):
    block: Block
    geometry: Polygon | MultiPolygon

    @model_validator(mode="before")
    @classmethod
    def validate_model(cls, data):
        # set service area as geometry's area
        area = data["geometry"].area
        data["area"] = area
        # if there is no data about capacity in service
        if not "capacity" in data:
            service_type = data["service_type"]
            bricks = service_type.get_bricks(is_integrated=False)
            brick = min(bricks, key=lambda x: abs(x.area - area))
            data["capacity"] = brick.capacity
        return data

    def to_dict(self) -> dict:
        return {"geometry": self.geometry, "block_id": self.block.id, **super().to_dict()}


class BuildingService(Service):
    building: Building
    geometry: Point

    @model_validator(mode="before")
    @classmethod
    def validate_model(cls, data):
        # set service's geometry as building's representative point (centroid)
        data["geometry"] = data["building"].geometry.representative_point()
        # if there is no data about area in service
        if not "area" in data:
            service_type = data["service_type"]
            bricks = service_type.get_bricks(is_integrated=True)
            # we try to find similar to our capacity
            if "capacity" in data:
                capacity = data["capacity"]
                brick = min(bricks, key=lambda x: abs(x.capacity - capacity))
                data["area"] = brick.area
            # or we try find the smallest possible capacity and area
            else:
                brick = min(bricks, key=lambda x: x.area)
                data["capacity"] = brick.capacity
                data["area"] = brick.area
        return data

    def to_dict(self) -> dict:
        return {
            "geometry": self.geometry,
            "block_id": self.building.block.id,
            "building_id": self.building.id,
            **super().to_dict(),
        }


class Building(BaseModel):
    id: int
    """Unique identifier across the ``City``"""
    model_config = ConfigDict(arbitrary_types_allowed=True, allow_inf_nan=False)
    block: Block
    """Parent block"""
    services: list[BuildingService] = []
    """List of services inside the building"""
    geometry: Polygon | MultiPolygon
    """Geometry representing a building"""
    build_floor_area: float = Field(ge=0)
    """Total area of the building in square meters"""
    living_area: float = Field(ge=0)
    """Building's area dedicated for living in square meters """
    business_area: float = Field(ge=0)
    """Building's area dedicated for non-living activities in square meters"""
    footprint_area: float = Field(ge=0)
    """Building's ground floor area in square meters """
    number_of_floors: int = Field(ge=1)
    """Number of floors (storeys) in the building"""
    population: int = Field(ge=0)
    """Total population of the building"""

    @property
    def is_living(self) -> bool:
        return self.living_area > 0

    def update_services(self, service_type: ServiceType, gdf: gpd.GeoDataFrame | None = None):
        """Update services of the building"""
        if gdf is None:
            self.services = list(filter(lambda s: s.service_type != service_type, self.services))
        else:
            services = [
                BuildingService(service_type=service_type, building=self, **gdf.loc[i].to_dict()) for i in gdf.index
            ]
            self.services = [*self.services, *services]

    def to_dict(self):
        return {
            "id": self.id,
            "block_id": self.block.id,
            "geometry": self.geometry,
            "population": self.population,
            "footprint_area": self.footprint_area,
            "build_floor_area": self.build_floor_area,
            "living_area": self.living_area,
            "business_area": self.business_area,
            "number_of_floors": self.number_of_floors,
            "is_living": self.is_living,
        }


class Block(BaseModel):
    """Class representing city block"""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    id: int
    """Unique block identifier across the ``City``"""
    geometry: Polygon
    """Block geometry presented as shapely ``Polygon``"""
    land_use: LandUse | None = None
    """Current city block landuse"""
    buildings: list[Building] = []
    """Buildings list inside of the block"""
    services: list[BlockService] = []
    """Services that take some area of the block"""
    city: City
    """``City`` instance that contains the block"""

    @field_validator("land_use", mode="before")
    def validate_land_use(value):
        if isinstance(value, str):
            value = value.lower()
            value = value.replace("-", "_")
            return value
        if isinstance(value, LandUse):
            return value
        return None

    @property
    def all_services(self) -> list[Service]:
        building_services = [s for b in self.buildings for s in b.services]
        return [*self.services, *building_services]

    @property
    def site_area(self):
        """Block area in square meters"""
        return self.geometry.area

    @property
    def population(self):
        """Block total population"""
        return sum([b.population for b in self.buildings], 0)

    @property
    def footprint_area(self):
        """Block total footprint area of the buildings"""
        return sum([b.footprint_area for b in self.buildings], 0)

    @property
    def build_floor_area(self):
        """Block total build floor area of the buildings"""
        return sum([b.build_floor_area for b in self.buildings], 0)

    @property
    def living_area(self):
        """Block total living area of the buildings"""
        return sum([b.living_area for b in self.buildings], 0)

    @property
    def business_area(self):
        """Block total non-living area of the buildings"""
        return sum([b.business_area for b in self.buildings], 0)

    @property
    def is_living(self):
        """Does a block contain any living building"""
        return any([b.is_living for b in self.buildings])

    @property
    def living_demand(self):
        """Square meters of living area per person"""
        try:
            return self.living_area / self.population
        except ZeroDivisionError:
            return None

    @property
    def fsi(self):
        """Floor space index (build floor area per site area)"""
        return self.build_floor_area / self.site_area

    @property
    def gsi(self):
        """Ground space index (footprint area per site area)"""
        return self.footprint_area / self.site_area

    @property
    def mxi(self):
        """Mixed use index (living area per build floor area)"""
        try:
            return self.living_area / self.build_floor_area
        except ZeroDivisionError:
            return None

    @property
    def l(self):
        """Mean number of floors"""
        try:
            return self.fsi / self.gsi
        except ZeroDivisionError:
            return None

    @property
    def osr(self):
        """Open space ratio"""
        try:
            return (1 - self.gsi) / self.fsi
        except ZeroDivisionError:
            return None

    @property
    def share_living(self):
        """Living area share"""
        try:
            return self.living_area / self.footprint_area
        except ZeroDivisionError:
            return None

    @property
    def share_business(self):
        """Business area share"""
        try:
            return self.business_area / self.footprint_area
        except ZeroDivisionError:
            return None

    @property
    def buildings_indicators(self):
        return {
            "build_floor_area": self.build_floor_area,
            "living_demand": self.living_demand,
            "living_area": self.living_area,
            "share_living": self.share_living,
            "business_area": self.business_area,
            "share_business": self.share_business,
        }

    @property
    def territory_indicators(self):
        return {
            "site_area": self.site_area,
            "population": self.population,
            "footprint_area": self.footprint_area,
            "fsi": self.fsi,
            "gsi": self.gsi,
            "l": self.l,
            "osr": self.osr,
            "mxi": self.mxi,
        }

    @property
    def services_indicators(self):
        service_types = dict.fromkeys([service.service_type for service in self.all_services], 0)

        return {
            f"capacity_{st.name}": sum(
                map(lambda s: s.capacity, filter(lambda s: s.service_type == st, self.all_services))
            )
            for st in service_types
        }

    @property
    def land_use_service_types(self) -> list[ServiceType]:
        return self.city.get_land_use_service_types(self.land_use)

    def get_services_gdf(self) -> gpd.GeoDataFrame:
        data = [service.to_dict() for service in self.all_services]
        return gpd.GeoDataFrame(data, crs=self.city.crs)

    def get_buildings_gdf(self) -> gpd.GeoDataFrame:
        data = [building.to_dict() for building in self.buildings]
        return gpd.GeoDataFrame(data, crs=self.city.crs).set_index("id")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "geometry": self.geometry,
            "land_use": None if self.land_use is None else self.land_use.value,
            "is_living": self.is_living,
            **self.buildings_indicators,
            **self.territory_indicators,
            **self.services_indicators,
        }

    def update_buildings(self, gdf: gpd.GeoDataFrame | None = None):
        """Update buildings GeoDataFrame of the block"""
        if gdf is None:
            self.buildings = []
        else:
            self.buildings = [Building(id=i, **gdf.loc[i].to_dict(), block=self) for i in gdf.index]

    def update_services(self, service_type: ServiceType, gdf: gpd.GeoDataFrame | None = None):
        """Update services of the block"""
        if gdf is None:
            self.services = list(filter(lambda s: s.service_type != service_type, self.services))
            for building in self.buildings:
                building.update_services(service_type)
        else:
            sjoin = gdf.sjoin(self.get_buildings_gdf()[["geometry"]], how="left")
            # for each building that has some services intersection we update it
            groups = sjoin.groupby("index_right")
            for building_id, services_gdf in groups:
                self[int(building_id)].update_services(service_type, services_gdf.drop(columns=["index_right"]))
            # for services that do not intersect anything we just update block services
            block_services_gdf = sjoin.loc[sjoin["index_right"].isna()]
            block_services = [
                BlockService(service_type=service_type, block=self, **block_services_gdf.loc[i].to_dict())
                for i in block_services_gdf.index
            ]
            self.services = [*self.services, *block_services]

    @classmethod
    def from_gdf(cls, gdf: gpd.GeoDataFrame, city: City) -> dict[int, Block]:
        """Generate blocks dict from ``GeoDataFrame``"""
        result = {}
        for i in gdf.index:
            result[i] = cls(id=i, geometry=gdf.loc[i].geometry, land_use=gdf.loc[i].land_use, city=city)
        return result

    @singledispatchmethod
    def __getitem__(self, arg):
        raise NotImplementedError(f"Can't access object with such argument type {type(arg)}")

    @__getitem__.register(int)
    def _(self, building_id: int) -> Building:
        """Make block subscriptable, to access building via id like ``block[123]``"""
        buildings_ids = [b.id for b in self.buildings]
        try:
            building_index = buildings_ids.index(building_id)
            building = self.buildings[building_index]
            return building
        except:
            raise KeyError(f"Can't find building with such id: {building_id}")

    def __hash__(self):
        """Make block hashable, so it can be used as key in dict etc."""
        return hash(self.id)


class City:
    """Block-network city model"""

    def __init__(self, blocks: gpd.GeoDataFrame, adj_mx: pd.DataFrame) -> None:
        assert (blocks.index == adj_mx.index).all(), "Matrix and blocks index don't match"
        assert (blocks.index == adj_mx.columns).all(), "Matrix columns and blocks index don't match"
        self.crs = blocks.crs
        self._blocks = Block.from_gdf(blocks, self)
        self.adjacency_matrix = adj_mx.copy()
        self._service_types = {}
        for st in SERVICE_TYPES:
            service_type = ServiceType(**st)
            self._service_types[service_type.name] = service_type

    @property
    def epsg(self):
        return self.crs.to_epsg()

    @property
    def blocks(self) -> list[Block]:
        """Return list of blocks"""
        return [b for b in self._blocks.values()]

    @property
    def service_types(self) -> list[ServiceType]:
        """Return list of service types"""
        return [st for st in self._service_types.values()]

    @property
    def buildings(self) -> list[Building]:
        """Return list of all buildings"""
        return [building for block in self.blocks for building in block.buildings]

    @property
    def services(self) -> list[Building]:
        """Return list of all services"""
        return [service for block in self.blocks for service in block.all_services]

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
                # legend=True,
                # legend_kwds={"title": "Service types", "loc": "lower left"},
            )
        ax.set_axis_off()

    def get_land_use_service_types(self, land_use: LandUse | None):
        filtered_service_types = filter(lambda st: land_use in st.land_use, self.service_types)
        return list(filtered_service_types)

    def get_buildings_gdf(self) -> gpd.GeoDataFrame | None:
        buildings = [b.to_dict() for b in self.buildings]
        return gpd.GeoDataFrame(buildings, crs=self.crs).set_index("id")

    def get_services_gdf(self) -> gpd.GeoDataFrame:
        services = [s.to_dict() for s in self.services]
        return gpd.GeoDataFrame(services, crs=self.crs)

    def get_blocks_gdf(self) -> gpd.GeoDataFrame:
        blocks = [b.to_dict() for b in self.blocks]
        return gpd.GeoDataFrame(blocks, crs=self.crs).set_index("id")

    def update_buildings(self, gdf: gpd.GeoDataFrame):
        """Update buildings in blocks"""
        assert gdf.crs == self.crs, "Buildings GeoDataFrame CRS should match city CRS"
        # reset buildings of blocks
        for block in self.blocks:
            block.update_buildings()
        # spatial join blocks and buildings and updated related blocks info
        sjoin = gdf.sjoin(self.get_blocks_gdf()[["geometry"]], how="left")
        groups = sjoin.groupby("index_right")
        for block_id, buildings_gdf in groups:
            self[int(block_id)].update_buildings(buildings_gdf)

    def update_services(self, service_type: ServiceType | str, gdf: gpd.GeoDataFrame):
        """Update services in blocks of certain service_type"""
        assert gdf.crs == self.crs, "Services GeoDataFrame CRS should match city CRS"
        service_type = self[service_type]
        # reset services of blocks
        for block in self.blocks:
            block.update_services(service_type)
        # spatial join blocks and services and update related blocks info
        sjoin = gdf.sjoin(self.get_blocks_gdf()[["geometry"]], how="left")
        groups = sjoin.groupby("index_right")
        for block_id, services_gdf in groups:
            self[int(block_id)].update_services(service_type, services_gdf.drop(columns=["index_right"]))

    def add_service_type(self, service_type: ServiceType):
        if service_type.name in self:
            raise KeyError(f"The service type with this name already exists: {service_type.name}")
        else:
            self._service_types[service_type.name] = service_type

    def get_distance(self, block_a: int | Block, block_b: int | Block):
        """Returns distance (in min) between two blocks"""
        return self[block_a, block_b]

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

    @__getitem__.register(Block)
    def _(self, block):
        """Placeholder for methods to avoid extra checks"""
        return block

    @__getitem__.register(int)
    def _(self, block_id):
        """Make city_model subscriptable, to access block via id like ``city[123]``"""
        if not block_id in self._blocks:
            raise KeyError(f"Can't find block with such id: {block_id}")
        return self._blocks[block_id]

    @__getitem__.register(ServiceType)
    def _(self, service_type):
        """Placeholder for methods to avoid extra checks"""
        return service_type

    @__getitem__.register(str)
    def _(self, service_type_name):
        """Make city_model subscriptable, to access service type via name like ``city['schools']``"""
        if not service_type_name in self._service_types:
            raise KeyError(f"Can't find service type with such name: {service_type_name}")
        return self._service_types[service_type_name]

    @__getitem__.register(tuple)
    def _(self, blocks):
        """Access distance between two blocks like ``city[block_a, block_b]``"""
        (block_a, block_b) = blocks
        block_a = self[block_a]
        block_b = self[block_b]
        return self.adjacency_matrix.loc[block_a.id, block_b.id]

    @singledispatchmethod
    def __contains__(self, arg):
        raise NotImplementedError(f"Wrong argument type for 'in': {type(arg)}")

    @__contains__.register(int)
    def _(self, block_id):
        """Make 'in' check available for blocks, to access like ``123 in city``"""
        return block_id in self._blocks

    @__contains__.register(str)
    def _(self, service_type_name):
        """Make 'in' check available for service types, to access like ``'schools' in city``"""
        return service_type_name in self._service_types

    def __str__(self):
        description = ""
        description += f"CRS : EPSG:{self.epsg}\n"
        description += f"Blocks : {len(self.blocks)}\n"
        description += f"Service types : {len(self.service_types)}\n"
        description += f"Buildings : {len(self.buildings)}\n"
        description += f"Services : {len(self.services)}\n"
        services_description = ""
        service_types = dict.fromkeys([service.service_type for service in self.services], 0)
        for service in self.services:
            service_types[service.service_type] += 1
        for service_type, count in service_types.items():
            services_description += f"    {service_type.name} : {count}\n"
        return description

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
