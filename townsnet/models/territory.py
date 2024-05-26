
import pyproj
import shapely
import geopandas as gpd
from .service_type import ServiceType
from pydantic import BaseModel, InstanceOf

class Territory(BaseModel):
  id : int
  name : str
  geometry : InstanceOf[shapely.Polygon]
  
  @staticmethod
  def _aggregate_provision(prov_gdf : gpd.GeoDataFrame):
      columns = ['demand', 'demand_left', 'demand_within', 'demand_without', 'capacity', 'capacity_left']
      result = {column:prov_gdf[column].sum() for column in columns}
      result['provision'] = result['demand_within'] / result['demand']
      return result

  def context_provision(self, service_type : ServiceType, towns_gdf : gpd.GeoDataFrame, speed : float = 283.33) -> tuple[gpd.GeoDataFrame, dict[str, float]]:
        
    buffer_meters = service_type.accessibility * speed
    
    towns_gdf = towns_gdf.copy()
    towns_gdf['distance'] = towns_gdf['geometry'].apply(lambda g : shapely.distance(g, self.geometry))
    towns_gdf = towns_gdf[towns_gdf['distance']<=buffer_meters]
    towns_gdf = towns_gdf.rename(columns={'index_right': 'town_id'})

    territory_dict = self._aggregate_provision(towns_gdf)
    territory_dict['buffer_meters'] = buffer_meters
    territory_dict['service_type'] = service_type.name
    return towns_gdf, territory_dict

  def to_dict(self):
    return {
      'id': self.id,
      'name': self.name,
      'geometry': self.geometry
    }

  @classmethod
  def from_gdf(cls, gdf : gpd.GeoDataFrame) -> list:
    return {i:cls(id=i, **gdf.loc[i].to_dict()) for i in gdf.index}