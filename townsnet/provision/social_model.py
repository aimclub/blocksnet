import geopandas as gpd
import pandas as pd
import pandera as pa
from shapely import Polygon, MultiPolygon, Point
from pandera.typing import Index, Series
from .service_type import ServiceType, AccessibilityType
from .schema import BaseSchema
from .provision_model import ProvisionModel

DEFAULT_TRAVEL_SPEED = 60 #km/h
DEFAULT_CRS = 4326
DISTANCE_COL = 'distance_meters'

class TownsSchema(BaseSchema):
  idx : Index[int] = pa.Field(unique=True)
  _geom_types = [Point]

  @pa.parser('geometry')
  @classmethod
  def parse_geometry(cls, series):
    name = series.name
    return series.representative_point().rename(name)

class ProvisionsSchema(pa.DataFrameModel):
  idx : Index[int] = pa.Field(unique=True)
  demand : Series[int] = pa.Field(ge=0, coerce=True) 
  demand_within : Series[int] = pa.Field(ge=0, coerce=True)

class SocialModel():

  def __init__(self, towns : gpd.GeoDataFrame, provisions : dict[ServiceType, pd.DataFrame], travel_speed : int = DEFAULT_TRAVEL_SPEED):
    self.towns : gpd.GeoDataFrame = TownsSchema(towns)
    # assert all(all(towns.index == provision.index) for provision in provisions.values())
    self.provisions = {st : ProvisionsSchema(prov) for st, prov in provisions.items()}
    self._travel_speed = travel_speed

  @property
  def estimated_crs(self):
    return self.towns.estimate_utm_crs()

  @property
  def travel_speed(self):
    # in meters per minute
    return self._travel_speed * 1_000 / 60

  def _get_towns_distances(self, project_gdf : gpd.GeoDataFrame):
    towns = self.towns.copy().to_crs(self.estimated_crs)
    return towns.sjoin_nearest(project_gdf, how='left', distance_col=DISTANCE_COL)[['geometry', DISTANCE_COL]]

  def _evaluate_provision(self, towns_distances : gpd.GeoDataFrame, service_type : ServiceType) -> float:
    if service_type.accessibility_type == AccessibilityType.METERS:
      accessibility_meters = service_type.accessibility_value
    else:
      accessibility_meters = service_type.accessibility_value * self.travel_speed

    context_towns = towns_distances[towns_distances[DISTANCE_COL]<=accessibility_meters]
    provision = self.provisions[service_type]
    provision = provision.loc[context_towns.index]
    
    if len(provision) == 0:
      return None

    return ProvisionModel.total(provision)

  def evaluate_provisions(self, project_geometry : Polygon | MultiPolygon) -> dict[ServiceType, float]:
    project_gdf = gpd.GeoDataFrame(geometry=[project_geometry], crs=DEFAULT_CRS).to_crs(self.estimated_crs)
    towns_distances = self._get_towns_distances(project_gdf)
    return {service_type: self._evaluate_provision(towns_distances, service_type) for service_type in self.provisions.keys()}

  def evaluate_social(self, provisions : dict[ServiceType, float]):
    provisions
    pd.DataFrame
  