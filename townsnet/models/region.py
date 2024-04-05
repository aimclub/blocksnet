from functools import singledispatchmethod
from pydantic import BaseModel, InstanceOf, field_validator, Field
from typing import Optional
import dill as pickle
import shapely
import geopandas as gpd
import pandas as pd
from .geodataframe import GeoDataFrame, BaseRow
from ..utils import SERVICE_TYPES
from .service_type import ServiceType

class RayonRow(BaseRow):
    name : str
    geometry : shapely.Polygon | shapely.MultiPolygon

class OkrugRow(BaseRow):
    name : str
    geometry : shapely.Polygon | shapely.MultiPolygon

class TownRow(BaseRow):
    name : str
    geometry : shapely.Point
    population : int

class ServiceRow(BaseRow):
    geometry : shapely.Point
    capacity : int = Field(gt=0)

class Town(BaseModel):
    id : int
    name : str
    population : int
    geometry : InstanceOf[shapely.Point]
    _capacities : dict[ServiceType, int] = {}

    def __contains__(self, service_type: ServiceType) -> bool:
        """Returns True if service type is contained in town"""
        return service_type in self._capacities

    def __getitem__(self, service_type: ServiceType) -> dict[str, int]:
        """Get service type capacity and demand of the town"""
        result = {"capacity": 0, "demand": service_type.calculate_in_need(self.population)}
        if service_type in self._capacities:
            result["capacity"] = self._capacities[service_type]
        return result

    def to_dict(self):
        res = {
            'id': self.id,
            'name': self.name,
            'population': self.population,
            'geometry': self.geometry
        }
        for service_type in self._capacities.keys():
            st_dict = self[service_type]
            for key, value in st_dict.items():
                res[f'{service_type.name}_{key}'] = value
        return res

    @classmethod
    def from_gdf(cls, gdf):
        res = {}
        for i in gdf.index:
            res[i] = cls(id=i, **gdf.loc[i].to_dict())
        return res 

    def update_capacity(self, service_type : ServiceType, value : int = 0):
        self._capacities[service_type] = value

class Region():

    def __init__(self, rayons : GeoDataFrame[RayonRow] | gpd.GeoDataFrame, okrugs : GeoDataFrame[OkrugRow] | gpd.GeoDataFrame, towns : GeoDataFrame[TownRow] | gpd.GeoDataFrame, adjacency_matrix : pd.DataFrame):
        if not isinstance(rayons, GeoDataFrame[RayonRow]):
            rayons = GeoDataFrame[RayonRow](rayons)
        if not isinstance(okrugs, GeoDataFrame[OkrugRow]):
            okrugs = GeoDataFrame[OkrugRow](okrugs)
        if not isinstance(towns, GeoDataFrame[TownRow]):
            towns = GeoDataFrame[TownRow](towns)
        assert (adjacency_matrix.index == adjacency_matrix.columns).all(), "Adjacency matrix indices and columns don't match"
        assert (adjacency_matrix.index == towns.index).all(), "Adjacency matrix indices and towns indices don't match"
        assert(rayons.crs == okrugs.crs and okrugs.crs == towns.crs), 'CRS should march everywhere'
        self.crs = towns.crs
        self.rayons = rayons
        self.okrugs = okrugs
        self.adjacency_matrix = adjacency_matrix
        self._towns = Town.from_gdf(towns)
        self._service_types = {}
        for st in SERVICE_TYPES:
            service_type = ServiceType(**st)
            self._service_types[service_type.name] = service_type

    def update_capacities(self, service_type: ServiceType | str, gdf: GeoDataFrame[ServiceRow]):
        """Update capacities in towns of certain service_type"""
        assert gdf.crs == self.crs, "Services GeoDataFrame CRS should match region CRS"
        if not isinstance(service_type, ServiceType):
            service_type = self[service_type]
        # reset services of towns
        for town in self.towns:
            town.update_capacity(service_type)
        # spatial join blocks and services and update related blocks info
        sjoin = gdf.sjoin_nearest(self.get_towns_gdf())
        groups = sjoin.groupby("index_right")
        for town_id, services_gdf in groups:
            self[town_id].update_capacity(service_type, services_gdf['capacity'].sum())

    def get_towns_gdf(self):
        towns = [town.to_dict() for town in self.towns]
        return gpd.GeoDataFrame(towns).set_index('id').set_crs(self.crs).fillna(0)

    def to_gdf(self):
        gdf = self.get_towns_gdf().sjoin(
            self.okrugs[['geometry', 'name']].rename(columns={'name': 'okrug_name'}), 
            how='left', 
            predicate='within',
            lsuffix='_town', 
            rsuffix='_okrug'
        )
        gdf = gdf.sjoin(
            self.rayons[['geometry', 'name']].rename(columns={'name':'rayon_name'}),
            how='left',
            predicate='within',
            lsuffix='_town',
            rsuffix='_rayon'
        )
        return gdf.drop(columns=['index__okrug', 'index__rayon'])

    @singledispatchmethod
    def __getitem__(self, arg):
        raise NotImplementedError(f"Can't access object with such argument type {type(arg)}")

    # Make city_model subscriptable, to access block via ID like city_model[123]
    @__getitem__.register(int)
    def _(self, town_id):
        if not town_id in self._towns:
            raise KeyError(f"Can't find town with such id: {town_id}")
        return self._towns[town_id]

    # Make city_model subscriptable, to access service type via name like city_model['schools']
    @__getitem__.register(str)
    def _(self, service_type_name):
        if not service_type_name in self._service_types:
            raise KeyError(f"Can't find service type with such name: {service_type_name}")
        return self._service_types[service_type_name]

    @__getitem__.register(tuple)
    def _(self, towns):
        (town_a, town_b) = towns
        if isinstance(town_a, Town):
            town_a = town_a.id
        if isinstance(town_b, Town):
            town_b = town_b.id
        return self.adjacency_matrix.loc[town_a, town_b]

    @property
    def towns(self):
        return self._towns.values()

    @property
    def service_types(self):
        return self._service_types.values()