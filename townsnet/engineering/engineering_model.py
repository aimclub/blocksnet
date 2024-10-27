import pandera as pa
import geopandas as gpd
from enum import Enum
from shapely import Point
from ..base_schema import BaseSchema

TERRITORY_INDEX_NAME = 'territory_id'

class EngineeringObject(Enum):
    ENGINEERING_OBJECT = 'Объект инженерной инфраструктуры'
    POWER_PLANTS = 'Электростанция'
    WATER_INTAKE = 'Водозабор'
    WATER_TREATMENT = 'Водоочистительное сооружение'
    WATER_RESERVOIR = 'Водохранилище'
    GAS_DISTRIBUTION = 'Газораспределительная станция'

class EngineeringsSchema(BaseSchema):
    ...
    # _geom_types = [Point]

    # @pa.parser('geometry')
    # @classmethod
    # def parse_geometry(cls, series):
    #     name = series.name
    #     return series.representative_point().rename(name)

class EngineeringModel():
  
    def __init__(self, gdfs : dict[EngineeringObject, gpd.GeoDataFrame]):
        self.gdfs = {eng_obj : EngineeringsSchema(gdf) for eng_obj, gdf in gdfs.items()}

    @staticmethod
    def _aggregate(gdf : gpd.GeoDataFrame, territories : gpd.GeoDataFrame):
        sjoin = gdf.sjoin(territories[['geometry']], predicate='intersects')
        return sjoin.groupby(TERRITORY_INDEX_NAME).size()

    def aggregate(self, territories : gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        territories = territories[['geometry']].copy()
        territories.index.name = TERRITORY_INDEX_NAME
        for eng_obj in list(EngineeringObject):
            if eng_obj in self.gdfs:
                gdf = self.gdfs[eng_obj]
                agg_gdf = self._aggregate(gdf, territories)
                territories[eng_obj.value] = agg_gdf
                territories[eng_obj.value] = territories[eng_obj.value].fillna(0).astype(int)
            else:
                territories[eng_obj.value] = 0
        return territories