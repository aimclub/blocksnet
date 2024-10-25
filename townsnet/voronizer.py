import pandera as pa
import pandas as pd
import geopandas as gpd
import shapely
from longsgis import voronoiDiagram4plg # pip install voronoi-diagram-for-polygons
from tqdm import tqdm
from .base_schema import BaseSchema

class SettlementsSchema(BaseSchema):
  _geom_types = [shapely.Point]

  @pa.parser('geometry')
  @classmethod
  def parse_geometry(cls, series):
    name = series.name
    return series.representative_point().rename(name)

class UnitsSchema(BaseSchema):
  _geom_types = [shapely.Polygon]

class Voronizer():

  def __init__(self, settlements_gdf : gpd.GeoDataFrame, units_gdf : gpd.GeoDataFrame):
    self.global_crs = settlements_gdf.crs
    self.local_crs = settlements_gdf.estimate_utm_crs()

    self.settlements_gdf = SettlementsSchema(settlements_gdf.to_crs(self.local_crs))
    units_gdf = units_gdf.explode(index_parts = True).reset_index().copy()
    self.units_gdf = UnitsSchema(units_gdf.to_crs(self.local_crs))

  @staticmethod
  def _voronize(points_gdf : gpd.GeoDataFrame, polygon_gdf : gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if len(points_gdf) == 1:
      return polygon_gdf.copy()
    vd = voronoiDiagram4plg(points_gdf[['geometry']], polygon_gdf[['geometry']])
    return vd[['geometry']]

  def run(self):
    #polygonize every unit settlements
    settlements_gdf = self.settlements_gdf[['geometry']].copy()
    units_gdf = self.units_gdf[['geometry']].copy()
    sjoin = settlements_gdf.sjoin(units_gdf, how='left', predicate='within')
    voronoi_gdfs = []
    for unit_i, unit_settlements_gdf in tqdm(sjoin.groupby('index_right')):
      unit_gdf = units_gdf[units_gdf.index == unit_i]
      voronoi_gdf = self._voronize(unit_settlements_gdf, unit_gdf)
      voronoi_gdfs.append(voronoi_gdf)
    #concat resulting gdf
    gdf = pd.concat(voronoi_gdfs).reset_index(drop=True)[['geometry']]
    sjoin = settlements_gdf.sjoin(gdf, predicate='within')
    sjoin.geometry = sjoin['index_right'].apply(lambda ir : gdf.loc[ir].geometry)
    return sjoin[['geometry']].to_crs(self.global_crs)