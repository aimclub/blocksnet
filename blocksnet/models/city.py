from __future__ import annotations
from abc import ABC
import math

import pickle
from functools import singledispatchmethod
from typing import Literal
from tqdm import tqdm
from loguru import logger
import geopandas as gpd
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from pydantic import BaseModel, Field, InstanceOf, field_validator, ConfigDict, model_validator, ValidationError
from shapely import Point, Polygon, MultiPolygon, LineString, intersection

from ..utils import SERVICE_TYPES
from .service_type import ServiceType, ServiceBrick
from .land_use import LandUse


class Service(ABC, BaseModel):
    """
    Abstract base class for services.

    Parameters
    ----------
    service_type : ServiceType
        Type of the service.
    capacity : int, optional
        Capacity of the service, must be greater than 0.
    area : float, optional
        Service area in square meters, must be greater than 0.
    is_integrated : bool
        Whether the service is integrated within a living building.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, allow_inf_nan=False)
    service_type: ServiceType
    capacity: int = Field(gt=0)
    area: float = Field(gt=0)
    is_integrated: bool

    @classmethod
    def _get_min_brick(
        cls, service_type: ServiceType, is_integrated: bool, field: Literal["area", "capacity"], value: float
    ) -> ServiceBrick:
        """
        Get the minimum service brick based on the service type and integration status.

        Parameters
        ----------
        service_type : ServiceType
            Type of the service.
        is_integrated : bool
            Whether the service is integrated within a living building.
        field : Literal["area", "capacity"]
            Field to compare, either "area" or "capacity".
        value : float
            Value to compare against.

        Returns
        -------
        Brick
            The brick with the minimum difference in the specified field value.
        """
        bricks = service_type.get_bricks(is_integrated)
        if len(bricks) == 0:
            bricks = service_type.get_bricks(not is_integrated)
        brick = min(bricks, key=lambda br: abs((br.area if field == "area" else br.capacity) - value))
        return brick

    @classmethod
    def _fill_capacity_and_area(cls, data: dict) -> dict:
        """
        Fill the capacity and area fields in the data dictionary.

        Parameters
        ----------
        data : dict
            Dictionary containing service data.

        Returns
        -------
        dict
            Updated dictionary with filled capacity and area fields.
        """
        data = data.copy()
        service_type = data["service_type"]
        is_integrated = data["is_integrated"]

        if "area" in data and not math.isnan(data["area"]):
            area = data["area"]
        else:
            area = data["geometry"].area

        if "capacity" in data and not math.isnan(data["capacity"]) and data["capacity"] > 0:
            capacity = data["capacity"]
            if area == 0:
                brick = cls._get_min_brick(service_type, is_integrated, "capacity", capacity)
                area = brick.area
        else:
            brick = cls._get_min_brick(service_type, is_integrated, "area", area)
            capacity = brick.capacity
            if area == 0:
                area = brick.area

        data.update({"area": area, "capacity": capacity})
        return data

    def to_dict(self) -> dict:
        """
        Convert the service to a dictionary representation.

        Returns
        -------
        dict
            Dictionary containing service data.
        """
        return {
            "service_type": self.service_type.name,
            "capacity": self.capacity,
            "area": self.area,
            "is_integrated": self.is_integrated,
        }


class BlockService(Service):
    """
    Service within a block taking some of a block's site area.

    Parameters
    ----------
    block : Block
        The block containing the service.
    geometry : Union[Point, Polygon, MultiPolygon]
        Geometry representing the service location.
    """

    block: Block
    geometry: Point | Polygon | MultiPolygon

    @model_validator(mode="before")
    @classmethod
    def validate_model(cls, data: dict) -> dict:
        """
        Validate and process the model data before initialization to make sure capacities and areas are filled.

        Parameters
        ----------
        data : dict
            Data dictionary containing service information.

        Returns
        -------
        dict
            Validated and processed data dictionary.
        """
        data["is_integrated"] = False
        data = cls._fill_capacity_and_area(data)
        return data

    def to_dict(self) -> dict:
        """
        Convert the block service to a dictionary representation.

        Returns
        -------
        dict
            Dictionary containing block service data.
        """
        return {"geometry": self.geometry, "block_id": self.block.id, **super().to_dict()}


class BuildingService(Service):
    """
    Service within a building.

    Parameters
    ----------
    building : Building
        The building containing the service.
    geometry : Point
        Point geometry representing the service location.
    """

    building: Building
    geometry: Point

    @model_validator(mode="before")
    @classmethod
    def validate_model(cls, data: dict) -> dict:
        """
        Validate and process the model data before initialization to figure out integration status and fill capacity and area if needed.

        Parameters
        ----------
        data : dict
            Data dictionary containing service information.

        Returns
        -------
        dict
            Validated and processed data dictionary.
        """
        data["is_integrated"] = data["building"].is_living
        data = cls._fill_capacity_and_area(data)
        return data

    @model_validator(mode="after")
    @classmethod
    def attach_geometry(cls, self) -> BuildingService:
        """
        Attach geometry of the service to the building representative point after initialization.

        Parameters
        ----------
        self : BuildingService
            Instance of the building service.

        Returns
        -------
        BuildingService
            The building service instance with attached geometry.
        """
        self.geometry = self.building.geometry.representative_point()
        return self

    def to_dict(self) -> dict:
        """
        Convert the building service to a dictionary representation.

        Returns
        -------
        dict
            Dictionary containing building service data.
        """
        return {
            "geometry": self.geometry,
            "block_id": self.building.block.id,
            "building_id": self.building.id,
            **super().to_dict(),
        }


class Building(BaseModel):
    """
    Represents a building within a block.

    Parameters
    ----------
    id : int
        Unique identifier across the whole city information model.
    block : Block
        Parent block containing the building.
    services : list of BuildingService, optional
        List of services inside the building.
    geometry : Union[Polygon, MultiPolygon]
        Geometry representing the building.
    build_floor_area : float
        Total area of the building in square meters. Must be equal or greater than 0.
    living_area : float
        Building's area dedicated for living in square meters. Must be equal or greater than 0.
    non_living_area : float
        Building's area dedicated for non-living activities in square meters. Must be equal or greater than 0.
    footprint_area : float
        Building's ground floor area in square meters. Must be equal or greater than 0.
    number_of_floors : int
        Number of floors (storeys) in the building. Must be equal or greater than 1.
    population : int
        Total population of the building. Must be equal or greater than 0.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, allow_inf_nan=False)
    id: int
    block: Block
    services: list[BuildingService] = []
    geometry: Polygon | MultiPolygon
    build_floor_area: float = Field(ge=0)
    living_area: float = Field(ge=0)
    non_living_area: float = Field(ge=0)
    footprint_area: float = Field(ge=0)
    number_of_floors: int = Field(ge=1)
    population: int = Field(ge=0)

    @model_validator(mode="before")
    @classmethod
    def validate_model(cls, data: dict) -> dict:
        """
        Validate and process the model data before initialization to fill needed empty fields.

        Parameters
        ----------
        data : dict
            Data dictionary containing building information.

        Returns
        -------
        dict
            Validated and processed data dictionary.
        """
        footprint_area = data["geometry"].area
        living_area = data["living_area"]

        if "build_floor_area" in data and not math.isnan(data["build_floor_area"]):
            build_floor_area = data["build_floor_area"]
            if "number_of_floors" in data and not math.isnan(data["number_of_floors"]):
                number_of_floors = data["number_of_floors"]
            else:
                number_of_floors = math.ceil(build_floor_area / footprint_area)
        else:
            if "number_of_floors" in data and not math.isnan(data["number_of_floors"]):
                number_of_floors = data["number_of_floors"]
                build_floor_area = number_of_floors * footprint_area
            else:
                raise ValueError("Either number_of_floors or build_floor_area should be defined")

        if "non_living_area" in data and not math.isnan(data["non_living_area"]):
            non_living_area = data["non_living_area"]
        else:
            non_living_area = build_floor_area - living_area

        data.update(
            {"footprint_area": footprint_area, "non_living_area": non_living_area, "build_floor_area": build_floor_area}
        )
        return data

    @property
    def is_living(self) -> bool:
        """Indicates if the building is residential."""
        return self.living_area > 0

    def update_services(self, service_type: ServiceType, gdf: gpd.GeoDataFrame | None = None) -> None:
        """
        Update services of the building.

        Parameters
        ----------
        service_type : ServiceType
            The type of service to be updated.
        gdf : GeoDataFrame, optional
            A GeoDataFrame containing service data. If not provided, services of specified type will be removed from the building.
        """
        if gdf is None:
            self.services = list(filter(lambda s: s.service_type != service_type, self.services))
        else:
            services = []
            for i in gdf.index:
                service = BuildingService(service_type=service_type, building=self, **gdf.loc[i].to_dict())
                services.append(service)
            self.services = [*self.services, *services]

    def to_dict(self) -> dict:
        """
        Convert the building object to a dictionary.

        Returns
        -------
        dict
            A dictionary representation of the building.
        """
        return {
            "id": self.id,
            "block_id": self.block.id,
            "geometry": self.geometry,
            "population": self.population,
            "footprint_area": self.footprint_area,
            "build_floor_area": self.build_floor_area,
            "living_area": self.living_area,
            "non_living_area": self.non_living_area,
            "number_of_floors": self.number_of_floors,
            "is_living": self.is_living,
        }


class Block(BaseModel):
    """
    Represents a city block.

    Parameters
    ----------
    id : int
        Unique block identifier across the whole city model.
    geometry : Polygon
        Geometry representing the city block.
    land_use : LandUse, optional
        Current city block land use.
    buildings : list of Building, optional
        List of buildings inside the block.
    services : list of BlockService, optional
        Services that occupy some area of the block.
    city : City
        Parent city instance that contains the block.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    id: int
    geometry: Polygon
    land_use: LandUse | None = None
    buildings: list[Building] = []
    services: list[BlockService] = []
    city: City

    @field_validator("land_use", mode="before")
    @staticmethod
    def validate_land_use(land_use: str | LandUse | None) -> LandUse | None:
        """
        Validate and transform the land use value.

        Parameters
        ----------
        land_use : str or LandUse
            The land use value to validate.

        Returns
        -------
        LandUse or None
            The validated land use.
        """
        if isinstance(land_use, str):
            land_use = land_use.lower()
            land_use = land_use.replace("-", "_")
            return land_use
        if isinstance(land_use, LandUse):
            return land_use
        return None

    @property
    def all_services(self) -> list[Service]:
        """
        Get all services in the block, including those in buildings.

        Returns
        -------
        list of Service
            List of all services in the block.
        """
        building_services = [s for b in self.buildings for s in b.services]
        return [*self.services, *building_services]

    @property
    def site_area(self) -> float:
        """
        Calculate the block area in square meters.

        Returns
        -------
        float
            Block area in square meters.
        """
        return self.geometry.area

    @property
    def population(self) -> int:
        """
        Calculate the total population of the block.

        Returns
        -------
        int
            Total population of the block.
        """
        return sum([b.population for b in self.buildings], 0)

    @property
    def footprint_area(self) -> float:
        """
        Calculate the total footprint area of the buildings in the block.

        Returns
        -------
        float
            Total footprint area of the buildings in square meters.
        """
        return sum([b.footprint_area for b in self.buildings], 0)

    @property
    def build_floor_area(self) -> float:
        """
        Calculate the total build floor area of the buildings in the block.

        Returns
        -------
        float
            Total build floor area of the buildings in square meters.
        """
        return sum([b.build_floor_area for b in self.buildings], 0)

    @property
    def living_area(self) -> float:
        """
        Calculate the total living area of the buildings in the block.

        Returns
        -------
        float
            Total living area of the buildings in square meters.
        """
        return sum([b.living_area for b in self.buildings], 0)

    @property
    def non_living_area(self) -> float:
        """
        Calculate the total non-living area of the buildings in the block.

        Returns
        -------
        float
            Total non-living area of the buildings in square meters.
        """
        return sum([b.non_living_area for b in self.buildings], 0)

    @property
    def is_living(self) -> bool:
        """
        Check if the block contains any living building and thus can be stated as living.

        Returns
        -------
        bool
            True if there is at least one living building, False otherwise.
        """
        return any([b.is_living for b in self.buildings])

    @property
    def living_demand(self) -> float | None:
        """
        Calculate the square meters of living area per person.

        Returns
        -------
        float or None
            Living area per person in square meters, or None if the population is 0.
        """
        try:
            return self.living_area / self.population
        except ZeroDivisionError:
            return None

    @property
    def fsi(self) -> float:
        """
        Calculate the Floor Space Index (FSI).

        Returns
        -------
        float
            FSI, which is the build floor area per site area.
        """
        return self.build_floor_area / self.site_area

    @property
    def gsi(self) -> float:
        """
        Calculate the Ground Space Index (GSI).

        Returns
        -------
        float
            GSI, which is the footprint area per site area.
        """
        return self.footprint_area / self.site_area

    @property
    def mxi(self) -> float | None:
        """
        Calculate the Mixed Use Index (MXI).

        Returns
        -------
        float or None
            MXI, which is the living area per build floor area, or None if the build floor area is 0.
        """
        try:
            return self.living_area / self.build_floor_area
        except ZeroDivisionError:
            return None

    @property
    def l(self) -> float | None:
        """
        Calculate the mean number of floors.

        Returns
        -------
        float or None
            Mean number of floors, or None if the GSI is 0.
        """
        try:
            return self.fsi / self.gsi
        except ZeroDivisionError:
            return None

    @property
    def osr(self) -> float | None:
        """
        Calculate the Open Space Ratio (OSR).

        Returns
        -------
        float or None
            OSR, or None if the FSI is 0.
        """
        try:
            return (1 - self.gsi) / self.fsi
        except ZeroDivisionError:
            return None

    @property
    def share_living(self) -> float | None:
        """
        Calculate the share of living area, which is living area per footprint area.

        Returns
        -------
        float or None
            Share of living area, or None if the footprint area is 0.
        """
        try:
            return self.living_area / self.footprint_area
        except ZeroDivisionError:
            return None

    @property
    def business_area(self) -> float:
        """
        Calculate the business area in the block, which is total business area of buildings inside the block.

        Returns
        -------
        float
            Business area in square meters.
        """
        return self.non_living_area

    @property
    def share_business(self) -> float | None:
        """
        Calculate the share of business area, which is business area per footprint area.

        Returns
        -------
        float or None
            Share of business area, or None if the footprint area is zero.
        """
        try:
            return self.business_area / self.footprint_area
        except ZeroDivisionError:
            return None

    @property
    def buildings_indicators(self) -> dict:
        """
        Get indicators related to buildings in the block.

        Returns
        -------
        dict
            Dictionary containing various building indicators.
        """
        return {
            "build_floor_area": self.build_floor_area,
            "living_demand": self.living_demand,
            "living_area": self.living_area,
            "share_living": self.share_living,
            "business_area": self.business_area,
            "share_business": self.share_business,
        }

    @property
    def territory_indicators(self) -> dict:
        """
        Get indicators related to the territory of the block.

        Returns
        -------
        dict
            Dictionary containing various territory indicators.
        """
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
    def services_indicators(self) -> dict:
        """
        Get indicators related to services in the block.

        Returns
        -------
        dict
            Dictionary containing service capacities by type.
        """
        service_types = dict.fromkeys([service.service_type for service in self.all_services], 0)

        return {
            f"capacity_{st.name}": sum(
                map(lambda s: s.capacity, filter(lambda s: s.service_type == st, self.all_services))
            )
            for st in service_types
        }

    @property
    def land_use_service_types(self) -> list[ServiceType]:
        """
        Get the service types allowed by the land use of the block.

        Returns
        -------
        list of ServiceType
            List of service types allowed by the land use.
        """
        return self.city.get_land_use_service_types(self.land_use)

    def get_services_gdf(self) -> gpd.GeoDataFrame:
        """
        Generate a GeoDataFrame of services in the block.

        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame containing services data.
        """
        data = [service.to_dict() for service in self.all_services]
        return gpd.GeoDataFrame(data, crs=self.city.crs)

    def get_buildings_gdf(self) -> gpd.GeoDataFrame:
        """
        Generate a GeoDataFrame of buildings in the block.

        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame containing buildings data.
        """
        data = [building.to_dict() for building in self.buildings]
        return gpd.GeoDataFrame(data, crs=self.city.crs).set_index("id")

    def to_dict(self, simplify=False) -> dict:
        """
        Convert the block to a dictionary representation.

        Parameters
        ----------
        simplify : bool, optional
            If True, exclude service indicators from the dictionary. Default is False.

        Returns
        -------
        dict
            Dictionary representation of the block.
        """
        res = {
            "id": self.id,
            "geometry": self.geometry,
            "land_use": None if self.land_use is None else self.land_use.value,
            "is_living": self.is_living,
            **self.buildings_indicators,
            **self.territory_indicators,
        }
        if not simplify:
            res = {**res, **self.services_indicators}
        return res

    def update_buildings(self, gdf: gpd.GeoDataFrame | None = None):
        """
        Update the buildings within the block.

        Parameters
        ----------
        gdf : gpd.GeoDataFrame, optional
            GeoDataFrame containing building data. If None, clear the list of buildings.
        """
        if gdf is None:
            self.buildings = []
        else:
            self.buildings = [Building(id=i, **gdf.loc[i].to_dict(), block=self) for i in gdf.index]

    def update_services(self, service_type: ServiceType, gdf: gpd.GeoDataFrame | None = None):
        """
        Update the services within the block.

        Parameters
        ----------
        service_type : ServiceType
            The type of service to update.
        gdf : gpd.GeoDataFrame, optional
            GeoDataFrame containing service data. If None, remove all services of the given type.
        """
        if gdf is None:
            self.services = list(filter(lambda s: s.service_type != service_type, self.services))
        else:
            services = []
            for i in gdf.index:
                service = BlockService(service_type=service_type, block=self, **gdf.loc[i].to_dict())
                services.append(service)
            self.services = [*self.services, *services]

    @classmethod
    def from_gdf(cls, gdf: gpd.GeoDataFrame, city: City) -> dict[int, Block]:
        """
        Generate a dictionary of blocks from a GeoDataFrame. Must contain following columns:
        - index : int
        - geometry : Polygon
        - land_use : LandUse or str

        For more specified information, please, check the Block class.

        Parameters
        ----------
        gdf : gpd.GeoDataFrame
            GeoDataFrame containing block data.
        city : City
            City instance containing the blocks.

        Returns
        -------
        dict of int : Block
            Dictionary mapping block IDs to Block instances.
        """
        result = {}
        for i in gdf.index:
            result[i] = cls(id=i, geometry=gdf.loc[i].geometry, land_use=gdf.loc[i].land_use, city=city)
        return result

    @singledispatchmethod
    def __getitem__(self, arg):
        """
        Access a building information or instance by a specified argument.

        Parameters
        ----------
        arg : Any
            Argument to access the building.

        Raises
        ------
        NotImplementedError
            If the argument type is not supported.
        """
        raise NotImplementedError(f"Can't access object with such argument type {type(arg)}")

    @__getitem__.register(int)
    def _(self, building_id: int) -> Building:
        """
        Access a building within the block by its ID.
        For example:

        >>> block = city[456]
        >>> block[123]
        Building(id=123, block_id=456, ...)

        Parameters
        ----------
        building_id : int
            ID of the building to access.

        Returns
        -------
        Building
            The building with the specified ID.

        Raises
        ------
        KeyError
            If no building with the specified ID is found.
        """
        buildings_ids = [b.id for b in self.buildings]
        try:
            building_index = buildings_ids.index(building_id)
            building = self.buildings[building_index]
            return building
        except:
            raise KeyError(f"Can't find building with such id: {building_id}")

    def __hash__(self):
        """
        Compute the hash value for the block.

        Returns
        -------
        int
            Hash value of the block.
        """
        return hash(self.id)


class City:
    """
    Represents a block-network city information model that manages blocks, buildings,
    services, and their relationships.

    Attributes
    ----------
    crs : pyproj.CRS
        Coordinate Reference System (CRS) used by the city model.
    adjacency_matrix : pd.DataFrame
        Adjacency matrix representing relationships between city blocks (travel time in minutes by drive, walk, intermodal or another type of city graph).
    _blocks : dict[int, Block]
        Dictionary mapping block IDs to Block objects.
    _service_types : dict[str, ServiceType]
        Dictionary mapping service type names to ServiceType objects.

    Methods
    -------
    plot(figsize: tuple = (15, 15), linewidth: float = 0.1, max_travel_time:float = 15)
        Plot the city model, displaying blocks, land use, buildings, and services.
    get_land_use_service_types(land_use: LandUse | None)
        Retrieve service types allowed by a specific land use.
    get_buildings_gdf() -> gpd.GeoDataFrame | None
        Return a GeoDataFrame of all buildings in the city model.
    get_services_gdf() -> gpd.GeoDataFrame
        Return a GeoDataFrame of all services in the city model.
    get_blocks_gdf(simplify=False) -> gpd.GeoDataFrame
        Return a GeoDataFrame of all blocks in the city model.
    update_land_use(gdf: gpd.GeoDataFrame) -> None
        Update land use of blocks based on a GeoDataFrame.
    update_buildings(gdf: gpd.GeoDataFrame) -> None
        Update buildings in the city model based on a GeoDataFrame.
    update_services(service_type: ServiceType | str, gdf: gpd.GeoDataFrame) -> None
        Update services of a specified type in the city model based on a GeoDataFrame.
    add_service_type(service_type: ServiceType) -> None
        Add a new service type to the city model.
    get_distance(block_a: int | Block, block_b: int | Block) -> float
        Get the distance (travel time) between two blocks.
    get_out_edges(block: int | Block)
        Get outgoing edges (connections) for a given block.
    get_in_edges(block: int | Block)
        Get incoming edges (connections) for a given block.
    """

    def __init__(self, blocks: gpd.GeoDataFrame, adj_mx: pd.DataFrame) -> None:
        """
        Initialize a City instance.

        Parameters
        ----------
        blocks : gpd.GeoDataFrame
            GeoDataFrame containing block data. Must contain following columns:
            - index : int
            - geometry : Polygon
            - land_use : LandUse or str

        adj_mx : pd.DataFrame
            DataFrame representing the adjacency matrix (or accessibility matrix). It should follow next rules:
            - the same index and columns as `blocks`.
            - values should contain travel times between row and column blocks as float.

        Raises
        ------
        AssertionError
            If the indices of `blocks` and `adj_mx` do not match.
        """
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
    def epsg(self) -> int:
        """
        Property to retrieve the EPSG code of the city model's CRS.

        Returns
        -------
        int
            EPSG code of the CRS.
        """
        return self.crs.to_epsg()

    @property
    def blocks(self) -> list[Block]:
        """
        Return a list of all blocks in the city model.

        Returns
        -------
        list[Block]
            List of Block objects representing the city blocks.
        """
        return [b for b in self._blocks.values()]

    @property
    def service_types(self) -> list[ServiceType]:
        """
        Return a list of all service types available in the city model.

        Returns
        -------
        list[ServiceType]
            List of ServiceType objects representing different service types.
        """
        return [st for st in self._service_types.values()]

    @property
    def buildings(self) -> list[Building]:
        """
        Return a list of all buildings in the city model.

        Returns
        -------
        list[Building]
            List of Building objects representing all buildings in the city.
        """
        return [building for block in self.blocks for building in block.buildings]

    @property
    def services(self) -> list[Service]:
        """
        Return a list of all services in the city model.

        Returns
        -------
        list[Service]
            List of Service objects representing all services available in the city.
        """
        return [service for block in self.blocks for service in block.all_services]

    def plot(
        self, figsize: tuple[float, float] = (15, 15), linewidth: float = 0.1, max_travel_time: float = 15
    ) -> None:
        """
        Plot the city model data including blocks, land use, buildings, and services.

        Parameters
        ----------
        figsize : tuple[float, float], optional
            Size of the plot to be displayed. Default is (15,15).
        linewidth : float, optional
            Line width of polygon objects to be displayed. Default is 0.1.
        max_travel_time : float, optional
            Max edge weight to be displayed on the plot (in min). Default is 15.
        """
        fig = plt.figure(figsize=figsize)
        grid = GridSpec(2, 2)

        ax_land_use = fig.add_subplot(grid[0, 0])
        ax_network = fig.add_subplot(grid[0, 1])
        ax_buildings = fig.add_subplot(grid[1, 0])
        ax_services = fig.add_subplot(grid[1, 1])
        blocks_gdf = self.get_blocks_gdf(True)

        for ax in [ax_land_use, ax_network, ax_buildings, ax_services]:
            blocks_gdf.plot(ax=ax, color="#777", linewidth=linewidth)
            ax.set_axis_off()

        # plot land_use
        ax_land_use.set_title("Land use")
        blocks_gdf.plot(ax=ax_land_use, linewidth=linewidth, column="land_use", legend=True)

        # plot network
        ax_network.set_title("Network")
        lines = []
        for i in self.adjacency_matrix.index:
            point_i = self[i].geometry.representative_point()
            series_i = self.adjacency_matrix.loc[i]
            for j in series_i[series_i <= max_travel_time].index:
                point_j = self[j].geometry.representative_point()
                lines.append(
                    {"geometry": LineString([point_i, point_j]), "travel_time": self.adjacency_matrix.loc[i, j]}
                )
        lines_gdf = gpd.GeoDataFrame(lines, crs=self.crs)
        lines_gdf.plot(ax=ax_network, linewidth=linewidth, column="travel_time", cmap="summer", legend=True)

        # plot buildings
        ax_buildings.set_title("Buildings")
        try:
            buildings_gdf = self.get_buildings_gdf()
            buildings_gdf.plot(ax=ax_buildings, linewidth=linewidth, column="population", legend=True, cmap="cool")
        except:
            ...

        # plot services
        ax_services.set_title("Services")
        try:
            services_gdf = self.get_services_gdf()
            services_gdf.plot(ax=ax_services, linewidth=linewidth, column="service_type", markersize=0.5)
        except:
            ...

    def get_land_use_service_types(self, land_use: LandUse | None) -> list[ServiceType]:
        """
        Retrieve service types allowed by a specific land use.

        Parameters
        ----------
        land_use : LandUse | None
            Land use type to filter service types by.

        Returns
        -------
        list[ServiceType]
            List of ServiceType objects associated with the specified land use.
        """
        filtered_service_types = filter(lambda st: land_use in st.land_use, self.service_types)
        return list(filtered_service_types)

    def get_buildings_gdf(self) -> gpd.GeoDataFrame:
        """
        Return a GeoDataFrame of all buildings in the city model.

        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame containing all buildings' geometries and attributes.
        """
        buildings = [b.to_dict() for b in self.buildings]
        return gpd.GeoDataFrame(buildings, crs=self.crs).set_index("id")

    def get_services_gdf(self) -> gpd.GeoDataFrame:
        """
        Return a GeoDataFrame of all services in the city model.

        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame containing all services' geometries and attributes.
        """
        services = [s.to_dict() for s in self.services]
        return gpd.GeoDataFrame(services, crs=self.crs)

    def get_blocks_gdf(self, simplify=False) -> gpd.GeoDataFrame:
        """
        Return a GeoDataFrame of all blocks in the city model.

        Parameters
        ----------
        simplify : bool, optional
            Whether to simplify block parameters (default is False). If True, services information will not be provided.

        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame containing all blocks' geometries and attributes.
        """
        blocks = [b.to_dict(simplify) for b in self.blocks]
        gdf = gpd.GeoDataFrame(blocks, crs=self.crs).set_index("id")
        if not simplify:
            for service_type in self.service_types:
                ...
                capacity_column = f"capacity_{service_type.name}"
                if not capacity_column in gdf.columns:
                    gdf[capacity_column] = 0
                else:
                    gdf[capacity_column] = gdf[capacity_column].fillna(0)
        return gdf

    def update_land_use(self, gdf: gpd.GeoDataFrame) -> None:
        """
        Update land use information for blocks based on a GeoDataFrame.

        Parameters
        ----------
        gdf : gpd.GeoDataFrame
            GeoDataFrame containing updated land use information. It should have the same CRS as the city model,
            and its index should match the IDs of blocks in the city model. It should have land_use column with value either LandUse, str or None.

        Raises
        ------
        AssertionError
            If the CRS of `gdf` does not match the city model's CRS, or if the index of `gdf` does not match block IDs,
            or if the length of `gdf` does not match the number of blocks in the city model.
        """
        assert gdf.crs == self.crs, "LandUse GeoDataFrame CRS should match city CRS"
        assert gdf.index.isin(self._blocks.keys()).all(), "Index should match blocks IDs"
        assert len(gdf) == len(self.blocks), "Index length must match blocks length"
        for i in gdf.index:
            land_use = gdf.loc[i, "land_use"]

            if isinstance(land_use, str):
                land_use = land_use.lower()
                land_use = land_use.replace("-", "_")
                land_use = LandUse(land_use)
            elif isinstance(land_use, LandUse):
                ...
            else:
                land_use = None
            self[i].land_use == land_use

    def update_buildings(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Update buildings information for blocks based on a GeoDataFrame.

        Parameters
        ----------
        gdf : gpd.GeoDataFrame
            GeoDataFrame containing updated buildings information. It should have the same CRS as the city model. It must contain the following columns:

            - index : int - unique building id
            - geometry : Polygon | MultiPolygon
            - build_floor_area : float >= 0
            - living_area : float >= 0
            - non_living_area : float >= 0
            - footprint_area : float >= 0
            - number_of_floors : int >= 1
            - population : int >= 0

            Please, do not specify building_id nor block_id columns, as they are used in the method. For more specific information, please, check Building class.

        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame buildings that did not intersect any block.

        Raises
        ------
        AssertionError
            If the CRS of `gdf` does not match the city model's CRS.

        """
        assert gdf.crs == self.crs, "Buildings GeoDataFrame CRS should match city CRS"
        # reset buildings of blocks
        logger.info("Removing existing blocks from the model")
        for block in self.blocks:
            block.update_buildings()
        # spatial join blocks and buildings and updated related blocks info
        logger.info("Joining buildings and blocks")
        sjoin = gdf.sjoin(self.get_blocks_gdf()[["geometry"]])
        sjoin = sjoin.rename(columns={"index_right": "block_id"})
        sjoin.geometry = sjoin.geometry.apply(
            lambda g: g.buffer(0) if g.geom_type in ["Polygon", "MultiPolygon"] else g
        )
        sjoin["intersection_area"] = sjoin.apply(
            lambda s: intersection(s.geometry, self[s.block_id].geometry), axis=1
        ).area
        sjoin["building_id"] = sjoin.index
        sjoin = sjoin.sort_values("intersection_area").drop_duplicates(subset="building_id", keep="last")
        if len(sjoin) < len(gdf):
            logger.warning(f"{len(gdf)-len(sjoin)} buildings did not intersect any block")
        groups = sjoin.groupby("block_id")
        for block_id, buildings_gdf in tqdm(groups, desc="Update blocks buildings"):
            self[int(block_id)].update_buildings(buildings_gdf)
        return gdf[~gdf.index.isin(sjoin.index)]

    def update_services(self, service_type: ServiceType | str, gdf: gpd.GeoDataFrame) -> None:
        """
        Update services information for blocks and buildings based on a GeoDataFrame.
        Based on intersections with city buildings, service may be either `BuldingService` or `BlockService`.
        Please, make sure, that block services (for example, pitch or playground) do not intersect any buildings!

        Parameters
        ----------
        service_type : ServiceType | str
            ServiceType object or name of the service type to update.
        gdf : gpd.GeoDataFrame
            GeoDataFrame containing updated services information. It should have the same CRS as the city model. It must contain the following columns:

            - index : int
            - geometry : Polygon | MultiPolygon | Point
            - capacity : int (optional)
            - area : int (optional)

            For more specified information, please, check Service class.

        Raises
        ------
        AssertionError
            If the CRS of `gdf` does not match the city model's CRS.
        """
        assert gdf.crs == self.crs, "Services GeoDataFrame CRS should match city CRS"
        service_type = self[service_type]

        # reset services of blocks and buildings
        for block in self.blocks:
            block.update_services(service_type)
        for building in self.buildings:
            building.update_services(service_type)

        # spatial join buildings and services and update related blocks info
        buildings_gdf = self.get_buildings_gdf()
        building_services = gdf.sjoin(buildings_gdf[["geometry", "block_id"]])
        building_services = building_services.rename(columns={"index_right": "building_id"})
        building_services.geometry = building_services.geometry.apply(
            lambda g: g.buffer(0) if g.geom_type in ["Polygon", "MultiPolygon"] else g
        )
        building_services["intersection_area"] = building_services.apply(
            lambda s: intersection(s.geometry, self[s.block_id][s.building_id].geometry), axis=1
        ).area
        building_services["service_id"] = building_services.index
        building_services = building_services.sort_values("intersection_area").drop_duplicates(
            subset="service_id", keep="last"
        )
        for building_info, services_gdf in building_services.groupby(["building_id", "block_id"]):
            building_id, block_id = building_info
            building = self[int(block_id)][int(building_id)]
            building.update_services(service_type, services_gdf)

        # spatial join block and rest of services
        blocks_gdf = self.get_blocks_gdf()
        block_services = gdf.loc[~gdf.index.isin(building_services.index)]
        block_services = block_services.sjoin(blocks_gdf[["geometry"]])
        block_services = block_services.rename(columns={"index_right": "block_id"})
        block_services.geometry = block_services.geometry.apply(
            lambda g: g.buffer(0) if g.geom_type in ["Polygon", "MultiPolygon"] else g
        )
        block_services["intersection_area"] = block_services.apply(
            lambda s: intersection(s.geometry, self[s.block_id].geometry), axis=1
        ).area
        block_services["service_id"] = block_services.index
        block_services = block_services.sort_values("intersection_area").drop_duplicates(
            subset="service_id", keep="last"
        )
        for block_id, gdf in block_services.groupby("block_id"):
            block = self[int(block_id)]
            block.update_services(service_type, gdf)

    def add_service_type(self, service_type: ServiceType) -> None:
        """
        Add a new service type to the city model.

        Parameters
        ----------
        service_type : ServiceType
            ServiceType object to be added.

        Raises
        ------
        KeyError
            If a service type with the same name already exists in the city model.
        """
        if service_type.name in self:
            raise KeyError(f"The service type with this name already exists: {service_type.name}")
        else:
            self._service_types[service_type.name] = service_type

    def get_distance(self, block_a: int | Block, block_b: int | Block) -> float:
        """
        Get the distance (in minutes) between two blocks.

        Parameters
        ----------
        block_a : int | Block
            ID or Block object representing the first block.
        block_b : int | Block
            ID or Block object representing the second block.

        Returns
        -------
        float
            Distance in minutes between the two blocks.
        """
        """Returns distance (in min) between two blocks"""
        return self[block_a, block_b]

    def get_out_edges(self, block: int | Block) -> list[(Block, Block, float)]:
        """
        Get outgoing edges for a specific block.

        Parameters
        ----------
        block : int | Block
            ID or Block object for which outgoing edges are requested.

        Returns
        -------
        list[(Block, Block, float)]
            List of tuples representing outgoing edges with weights between blocks.
        """
        if isinstance(block, Block):
            block = block.id
        return [(self[block], self[block_b], weight) for block_b, weight in self.adjacency_matrix.loc[block].items()]

    def get_in_edges(self, block: int | Block) -> list[(Block, Block, float)]:
        """
        Get incoming edges for a specific block.

        Parameters
        ----------
        block : int | Block
            ID or Block object for which incoming edges are requested.

        Returns
        -------
        list[(Block, Block, float)]
            List of tuples representing incoming edges with weights between blocks.
        """
        if isinstance(block, Block):
            block = block.id
        return [(self[block_b], self[block], weight) for block_b, weight in self.adjacency_matrix.loc[:, block].items()]

    @singledispatchmethod
    def __getitem__(self, arg):
        raise NotImplementedError(f"Can't access object with such argument type {type(arg)}")

    @__getitem__.register(Block)
    def _(self, block: Block) -> Block:
        """
        Placeholder for some methods to exclude type checks.

        Parameters
        ----------
        block : Block
            Block object to retrieve.

        Returns
        -------
        Block
            The requested Block object.
        """
        return block

    @__getitem__.register(int)
    def _(self, block_id: int) -> Block:
        """
        Get a block object by its ID. For example:
        >>> city[123]
        Block(id=123, ...)

        Parameters
        ----------
        block_id : int
            ID of the block to retrieve.

        Returns
        -------
        Block
            The requested Block object.

        Raises
        ------
        KeyError
            If the block with the specified ID does not exist in the city model.
        """
        if not block_id in self._blocks:
            raise KeyError(f"Can't find block with such id: {block_id}")
        return self._blocks[block_id]

    @__getitem__.register(ServiceType)
    def _(self, service_type: ServiceType) -> ServiceType:
        """
        Placeholder for some methods to exclude type checks.

        Parameters
        ----------
        service_type : ServiceType
            ServiceType object to retrieve.

        Returns
        -------
        ServiceType
            The requested ServiceType object.
        """
        return service_type

    @__getitem__.register(str)
    def _(self, service_type_name: str) -> ServiceType:
        """
        Get a service type object by its name. For example:
        >>> city['school']
        ServiceType(name='school', ...)

        Parameters
        ----------
        service_type_name : str
            Name of the service type to retrieve.

        Returns
        -------
        ServiceType
            The requested ServiceType object.

        Raises
        ------
        KeyError
            If a service type with the specified name does not exist in the city model.
        """
        if not service_type_name in self._service_types:
            raise KeyError(f"Can't find service type with such name: {service_type_name}")
        return self._service_types[service_type_name]

    @__getitem__.register(tuple)
    def _(self, blocks) -> float:
        """
        Get the distance (travel time in minutes) between two blocks. For example:

        >>> city[123, 456]
        34.5677

        >>> block_a = city[123]
        >>> block_b = city[456]
        >>> city[block_a, block_b]
        34.5677

        Parameters
        ----------
        blocks : tuple
            Tuple containing IDs or Block objects representing the two blocks.

        Returns
        -------
        float
            Distance in minutes between the two blocks.
        """
        (block_a, block_b) = blocks
        block_a = self[block_a]
        block_b = self[block_b]
        return self.adjacency_matrix.loc[block_a.id, block_b.id]

    @singledispatchmethod
    def __contains__(self, arg):
        raise NotImplementedError(f"Wrong argument type for 'in': {type(arg)}")

    @__contains__.register(int)
    def _(self, block_id: int) -> bool:
        """
        Check if a block exists in the city model. For example:
        >>> 123 in city
        True
        >>> 10000 in city
        False

        Parameters
        ----------
        block_id : int
            ID of the block to check.

        Returns
        -------
        bool
            True if the block exists in the city model, False otherwise.
        """
        return block_id in self._blocks

    @__contains__.register(str)
    def _(self, service_type_name: str) -> bool:
        """
        Check if a service type exists in the city model. For example:

        >>> 'school' in city
        True
        >>> 'computer_club' in city
        False

        Parameters
        ----------
        service_type_name : str
            Name of the service type to check.

        Returns
        -------
        bool
            True if the service type exists in the city model, False otherwise.
        """
        return service_type_name in self._service_types

    def __str__(self) -> str:
        """
        Return a string representation of the city model.

        Returns
        -------
        str
            String describing the city model, including CRS, number of blocks, service types, buildings, and services.
        """
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
        """
        Load a city model from a `.pickle` file.

        Parameters
        ----------
        file_path : str
            Path to the `.pickle` file containing the city model.

        Returns
        -------
        City
            The loaded City object from the `.pickle` file.
        """
        state = None
        with open(file_path, "rb") as f:
            state = pickle.load(f)
        return state

    def to_pickle(self, file_path: str):
        """
        Save the city model to a `.pickle` file.

        Parameters
        ----------
        file_path : str
            Path to the `.pickle` file where the city model will be saved.
        """
        with open(file_path, "wb") as f:
            pickle.dump(self, f)
