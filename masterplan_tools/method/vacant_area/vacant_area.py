
import osmnx as ox
import pandas as pd
import geopandas as gpd
from typing import ClassVar


from pydantic import BaseModel
from ...models import CityModel


class VacantArea(BaseModel):

    city_model: CityModel
 
    local_crs: ClassVar[int] = 32636
    roads_buffer: ClassVar[int] = 10
    buildings_buffer: ClassVar[int] = 10

    @classmethod
    def dwn_other(cls, block, local_crs):
        other = ox.geometries_from_polygon(block, tags={"man_made": True, "aeroway": True,"military": True })
        other['geometry'] = other['geometry'].to_crs(local_crs)
        return other.geometry
    
    @classmethod
    def dwn_leisure(cls, block, local_crs):
        leisure = ox.geometries_from_polygon(block, tags={"leisure": True})
        leisure['geometry'] = leisure['geometry'].to_crs(local_crs)
        return leisure.geometry
    
    @classmethod
    def dwn_landuse(cls,block, local_crs):
        landuse = ox.geometries_from_polygon(block, tags={"landuse": True})
        filtered_landuse = landuse[landuse['landuse'] != 'residential']
        filtered_landuse['geometry'] = filtered_landuse['geometry'].to_crs(local_crs)
        return filtered_landuse.geometry
    
    @classmethod
    def dwn_amenity(cls,block, local_crs):
        amenity = ox.geometries_from_polygon(block, tags={"amenity": True})
        amenity['geometry'] = amenity['geometry'].to_crs(local_crs)
        return amenity.geometry
    
    @classmethod
    def dwn_buildings(cls,block, local_crs ,buildings_buffer):
        buildings = ox.geometries_from_polygon(block, tags={"building": True})
        if buildings_buffer:
            buildings['geometry'] = buildings['geometry'].to_crs(local_crs).buffer(buildings_buffer)
        return buildings.geometry
    
    @classmethod
    def dwn_natural(cls,block, local_crs):
        natural = ox.geometries_from_polygon(block, tags={"natural": True})
        filtered_natural = natural[natural['natural'] != 'bay']
        filtered_natural['geometry'] = filtered_natural['geometry'].to_crs(local_crs)
        return filtered_natural.geometry
    
    @classmethod
    def dwn_waterway(cls,block, local_crs):
        waterway = ox.geometries_from_polygon(block, tags={"waterway": True})
        waterway['geometry'] = waterway['geometry'].to_crs(local_crs)
        return waterway.geometry
    
    @classmethod
    def dwn_highway(cls, block, local_crs, roads_buffer):
        highway = ox.geometries_from_polygon(block, tags={"highway": True})
        condition = (highway['highway'] != 'path') & (highway['highway'] != 'footway') & (highway['highway'] != 'pedestrian')
        filtered_highway = highway[condition]
        if roads_buffer is not None:
            filtered_highway['geometry'] = filtered_highway['geometry'].to_crs(local_crs).buffer(roads_buffer)
        filtered_highway['geometry'] = filtered_highway['geometry'].to_crs(local_crs).buffer(1)
        return filtered_highway.geometry
    
    @classmethod
    def dwn_path(cls, block, local_crs):
        tags = {'highway': 'path', 'highway': 'footway'}
        path = ox.geometries_from_polygon(block, tags=tags)
        path['geometry'] = path['geometry'].to_crs(local_crs).buffer(0.5)
        return path.geometry
    
    @classmethod
    def dwn_railway(cls, block, local_crs):
        railway = ox.geometries_from_polygon(block, tags={"railway": True})
        filtered_railway = railway[railway['railway'] != 'subway']
        filtered_railway['geometry'] = filtered_railway['geometry'].to_crs(local_crs)
        return filtered_railway.geometry
    
    @classmethod
    def create_minimum_bounding_rectangle(cls, polygon):
        return polygon.minimum_rotated_rectangle
    @classmethod
    def buffer_and_union(cls, row, buffer_distance=1):
        polygon = row['geometry']
        buffer_polygon = polygon.buffer(buffer_distance)
        return buffer_polygon
    
    def get_vacant_area(self, blpck_id:int):
        blocks= self.city_model.blocks.to_gdf().copy()
        blocks = gpd.GeoDataFrame(geometry=gpd.GeoSeries(blocks.geometry))
        if blpck_id:
            block_gdf = gpd.GeoDataFrame([blocks.iloc[blpck_id]], crs=blocks.crs)
            block_buffer = block_gdf['geometry'].buffer(20).to_crs(epsg=4326).iloc[0]
        else:
            block_gdf = blocks
            block_buffer = blocks.buffer(20).to_crs(epsg=4326).unary_union

        leisure = self.dwn_leisure(block_buffer, self.local_crs)
        landuse = self.dwn_landuse(block_buffer, self.local_crs)
        other  = self.dwn_other(block_buffer, self.local_crs)
        amenity = self.dwn_amenity(block_buffer, self.local_crs)
        buildings = self.dwn_buildings(block_buffer, self.local_crs, self.buildings_buffer)
        natural = self.dwn_natural(block_buffer, self.local_crs)
        waterway = self.dwn_waterway(block_buffer, self.local_crs)
        highway = self.dwn_highway(block_buffer, self.local_crs, self.roads_buffer)
        railway = self.dwn_railway(block_buffer, self.local_crs)
        path = self.dwn_path(block_buffer, self.local_crs)

        occupied_area = [leisure, other, landuse, amenity, buildings, natural, waterway, highway, railway, path]
        occupied_area = pd.concat(occupied_area)
        occupied_area = gpd.GeoDataFrame(geometry=gpd.GeoSeries(occupied_area))

        polygon = occupied_area.geometry.geom_type == "Polygon"
        multipolygon = occupied_area.geometry.geom_type == "MultiPolygon"
        blocks_new = gpd.overlay(block_gdf, occupied_area[polygon], how="difference")
        blocks_new = gpd.overlay(blocks_new, occupied_area[multipolygon], how="difference")
        blocks_exploded = blocks_new.explode(index_parts=True)
        blocks_exploded.reset_index(drop=True,inplace=True)

        blocks_filtered_area = blocks_exploded[blocks_exploded['geometry'].area >= 100] #1
        area_attitude = 1.9 # 2
        for index, row in blocks_filtered_area.iterrows():
            polygon = row['geometry']
            mbr = self.create_minimum_bounding_rectangle(polygon)
            if polygon.area * area_attitude < mbr.area:
                blocks_filtered_area.drop(index, inplace=True)

        blocks_filtered_area['buffered_geometry'] = blocks_filtered_area.apply(self.buffer_and_union, axis=1)
        unified_geometry = blocks_filtered_area['buffered_geometry'].unary_union

        result_gdf = gpd.GeoDataFrame(geometry=[unified_geometry], crs=blocks_filtered_area.crs)
        result_gdf_exploded = result_gdf.explode(index_parts=True)
        result_gdf_exploded['area'] = result_gdf_exploded['geometry'].area
        result_gdf_exploded.reset_index(drop=True, inplace=True)
        return result_gdf_exploded
