import geopandas as gpd
import json
import h3
from shapely import to_geojson, Polygon, MultiPolygon

DEFAULT_RESOLUTION = 6

class GridGenerator():

  def __init__(self, resolution : int = DEFAULT_RESOLUTION):
    self.resolution = resolution

  @property
  def cell_size(self) -> float:
    return h3.hex_area(self.resolution, 'km^2')*100

  def run(self, gdf : gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    
    polygon = gdf.unary_union    
    geojson = json.loads(to_geojson(polygon.convex_hull))

    hexagons = h3.polyfill(geojson, self.resolution, True)
    hexagons_geom = [Polygon(h3.h3_to_geo_boundary(h, geo_json=True)) for h in hexagons]

    hexgrid = gpd.GeoDataFrame(geometry=hexagons_geom, crs=gdf.crs)
    hexgrid = hexgrid.sjoin(gdf, predicate='within').reset_index()[['geometry']]

    return hexgrid