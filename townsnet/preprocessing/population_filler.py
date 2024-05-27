from statistics import median
from pydantic import BaseModel, model_validator, field_validator, Field, InstanceOf
from ..models.geodataframe import GeoDataFrame, BaseRow
import shapely
import pandas as pd

class UnitRow(BaseRow):
  geometry : shapely.Polygon | shapely.MultiPolygon
  population : int

class TownRow(BaseRow):
  geometry : shapely.Point
  name : str
  is_city : bool

class PopulationFiller(BaseModel):
  units : GeoDataFrame[UnitRow]
  towns : GeoDataFrame[TownRow]
  adjacency_matrix : InstanceOf[pd.DataFrame]
  city_multiplier : float = Field(gt=0, default=10)

  @field_validator('units', mode='before')
  @classmethod
  def validate_units(cls, gdf):
    if not isinstance(gdf, GeoDataFrame[UnitRow]):
      gdf = GeoDataFrame[UnitRow](gdf)
    return gdf

  @field_validator('towns', mode='before')
  @classmethod
  def validate_towns(cls, gdf):
    if not isinstance(gdf, GeoDataFrame[TownRow]):
      gdf = GeoDataFrame[TownRow](gdf)
    return gdf

  @field_validator('adjacency_matrix', mode='after')
  @classmethod
  def validate_adjacency_matrix(cls, df):
    assert all(df.index == df.columns), "Matrix index and columns don't match"
    return df

  @model_validator(mode='after')
  def validate_model(self):
    adj_mx = self.adjacency_matrix
    towns = self.towns.copy()
    units = self.units
    assert towns.crs == units.crs, "Units and towns CRS don't match"
    assert all(adj_mx.index == towns.index), "Matrix index and towns index don't match"
    return self

  def _get_median_time(self, town_id):
    return median(self.adjacency_matrix.loc[town_id])

  def fill(self):
    towns = self.towns.copy()
    towns['median_time'] = towns.apply(lambda x : self._get_median_time(x.name), axis=1)
    for i in self.units.index:
      geometry = self.units.loc[i, 'geometry']
      population = self.units.loc[i, 'population']
      unit_towns = towns.loc[towns.within(geometry)].copy()
      unit_towns['coef'] = unit_towns.apply(lambda x : (self.city_multiplier if x['is_city'] else 1)/x['median_time'], axis=1)
      coef_sum = unit_towns['coef'].sum()
      unit_towns['coef_norm'] = unit_towns['coef'] / coef_sum
      unit_towns['population'] = population * unit_towns['coef_norm']
      for j in unit_towns.index:
        towns.loc[j, 'coef'] = unit_towns.loc[j, 'coef']
        towns.loc[j, 'coef_norm'] = unit_towns.loc[j, 'coef_norm']
        towns.loc[j,'population'] = round(unit_towns.loc[j,'population'])
    return towns