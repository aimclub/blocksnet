import pandera as pa
import geopandas as gpd
from enum import Enum
from shapely import Point
from ..base_schema import BaseSchema

class EngineeringObject(Enum):
  ENGINEERING_OBJECT = 'Объект инженерной инфраструктуры'
  POWER_PLANTS = 'Электростанция'
  WATER_INTAKE = 'Водозабор'
  WATER_TREATMENT = 'Водоочистительное сооружение'
  WATER_RESERVOIR = 'Водохранилище'
  GAS_DISTRIBUTION = 'Газораспределительная станция'

class EngineeringsSchema(BaseSchema):
  _geom_types = [Point]

  @pa.parser('geometry')
  @classmethod
  def parse_geometry(cls, series):
    name = series.name
    return series.representative_point().rename(name)

class EngineeringModel():
  
  def __init__(self, gdfs : dict[EngineeringObject, gpd.GeoDataFrame]):
    self.gdfs = {eng_obj : EngineeringsSchema(gdf) for eng_obj, gdf in gdfs.items()}

  @staticmethod
  def _aggregate(gdf : gpd.GeoDataFrame, units : gpd.GeoDataFrame):
    sjoin = gdf.sjoin(units[['geometry']], predicate='within')
    return sjoin.groupby('index_right').size()

  def aggregate(self, units : gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    units = units[['geometry']].copy()
    for eng_obj in list(EngineeringObject):
      if eng_obj in self.gdfs:
        gdf = self.gdfs[eng_obj]
        agg_gdf = self._aggregate(gdf, units)
        units[eng_obj.value] = agg_gdf
        units[eng_obj.value] = units[eng_obj.value].fillna(0).astype(int)
      else:
        units[eng_obj.value] = 0
    return units