from functools import singledispatchmethod
from pydantic import BaseModel, InstanceOf, field_validator, Field
from typing import Optional
import dill as pickle
import pyproj
import shapely
import geopandas as gpd
import pandas as pd
from ..utils.basic_service_types import BASIC_SERVICE_TYPES
from .service_type import ServiceType, ServiceCategory
from .town import Town, Service

class Region():
    
    def __init__(self, districts : gpd.GeoDataFrame, settlements : gpd.GeoDataFrame, towns : gpd.GeoDataFrame, accessibility_matrix : pd.DataFrame):
        
        districts = self.validate_districts(districts)
        settlements = self.validate_settlements(settlements)
        towns = self.validate_towns(towns)
        accessibility_matrix = self.validate_accessibility_matrix(accessibility_matrix)

        assert (accessibility_matrix.index == towns.index).all(), "Accessibility matrix indices and towns indices don't match"
        assert(districts.crs == settlements.crs == towns.crs), 'CRS should match everywhere'

        self.crs = towns.crs
        self.districts = districts
        self.settlements = settlements
        self._towns = Town.from_gdf(towns)
        self.accessibility_matrix = accessibility_matrix

        service_types = {}
        for infrastructure, service_types_dicts in BASIC_SERVICE_TYPES.items():
            for service_type_dict in service_types_dicts:
                service_types[service_type_dict['name']] = ServiceType(category=ServiceCategory.BASIC, infrastructure=infrastructure, **service_type_dict)
        self._service_types = service_types

    @staticmethod
    def validate_districts(gdf : gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        assert isinstance(gdf, gpd.GeoDataFrame), 'Districts should be instance of gpd.GeoDataFrame'
        assert gdf.geom_type.isin(['Polygon', 'MultiPolygon']).all(), 'District geometry should be Polygon or MultiPolygon'
        assert pd.api.types.is_string_dtype(gdf['name']), 'District name should be str'
        return gdf[['geometry', 'name']]

    @staticmethod
    def validate_settlements(gdf : gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        assert isinstance(gdf, gpd.GeoDataFrame), 'Settlements should be instance of gpd.GeoDataFrame'
        assert gdf.geom_type.isin(['Polygon', 'MultiPolygon']).all(), 'Settlement geometry should be Polygon or MultiPolygon'
        assert pd.api.types.is_string_dtype(gdf['name']), 'Settlement name should be str'
        return gdf[['geometry', 'name']]
    
    @staticmethod
    def validate_towns(gdf : gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        assert isinstance(gdf, gpd.GeoDataFrame), 'Towns should be instance of gpd.GeoDataFrame'
        return gdf[['geometry', 'name', 'population']]
    
    @staticmethod
    def validate_accessibility_matrix(df : pd.DataFrame) -> pd.DataFrame:
        assert pd.api.types.is_float_dtype(df.values), 'Accessibility matrix values should be float'
        assert (df.values>=0).all(), 'Accessibility matrix values should be greater or equal 0'
        assert (df.index == df.columns).all(), "Accessibility matrix indices and columns don't match"
        return df
    
    @staticmethod
    def validate_services(gdf : gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        assert isinstance(gdf, gpd.GeoDataFrame), 'Services should be instance of gpd.GeoDataFrame'
        return gdf[['geometry', 'town_id', 'capacity']]

    @property
    def towns(self) -> list[Town]:
        return self._towns.values()

    @property
    def service_types(self) -> list[ServiceType]:
        return self._service_types.values()
    
    @property
    def geometry(self) -> shapely.Polygon | shapely.MultiPolygon:
        return self.districts.to_crs(4326).unary_union
    
    def match_services_towns(self, gdf : gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        assert gdf.crs == self.crs, 'Services GeoDataFrame CRS should match region CRS'
        gdf = gdf.copy()
        towns_gdf = self.get_towns_gdf()[['geometry', 'population']]
        
        def get_closest_city(service_i):
            service_gdf = gdf[gdf.index == service_i]
            sjoin = towns_gdf.sjoin_nearest(service_gdf, distance_col='distance')
            sjoin['weight'] = sjoin['population'] / sjoin['distance'] / sjoin['distance']
            return sjoin['weight'].idxmax()
        
        gdf['town_id'] = gdf.apply(lambda s : get_closest_city(s.name), axis=1)
        return gdf


    def update_services(self, service_type: ServiceType | str, gdf: gpd.GeoDataFrame):
        """Update capacities in towns of certain service_type"""
        gdf = self.validate_services(gdf)
        assert gdf.crs == self.crs, "Services GeoDataFrame CRS should match region CRS"
        if not isinstance(service_type, ServiceType):
            service_type = self[service_type]
        # reset services of towns
        for town in self.towns:
            town.update_services(service_type)
        # spatial join blocks and services and update related blocks info
        groups = gdf.groupby("town_id")
        for town_id, services_gdf in groups:
            self[town_id].update_services(service_type, services_gdf)

    def get_services_gdf(self) -> gpd.GeoDataFrame:
        data = [{'town':town.name, **service.to_dict()} for town in self.towns for service in town.services]
        return gpd.GeoDataFrame(data, crs=self.crs)

    def get_towns_gdf(self) -> gpd.GeoDataFrame:
        data = [town.to_dict() for town in self.towns]
        gdf = gpd.GeoDataFrame(data, crs=self.crs).rename(columns={'name': 'town_name'}).set_index('id', drop=True)
        gdf = gdf.sjoin(
            self.settlements[['geometry', 'name']].rename(columns={'name': 'settlement_name'}), 
            how='left',
            predicate='within',
            lsuffix='town',
            rsuffix='settlement'
        )
        gdf = gdf.sjoin(
            self.districts[['geometry', 'name']].rename(columns={'name':'district_name'}),
            how='left',
            predicate='within',
            lsuffix='town',
            rsuffix='district'
        )
        return gdf.drop(columns=['index_settlement', 'index_district']).fillna(0)
    
    def get_service_types_df(self) -> pd.DataFrame:
        data = [service_type.to_dict() for service_type in self.service_types]
        return pd.DataFrame(data)

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
        return self.accessibility_matrix.loc[town_a, town_b]
    
    @staticmethod
    def from_pickle(file_path: str):
        """Load region model from a .pickle file"""
        state = None
        with open(file_path, "rb") as f:
            state = pickle.load(f)
        return state

    def to_pickle(self, file_path: str):
        """Save region model to a .pickle file"""
        with open(file_path, "wb") as f:
            pickle.dump(self, f)