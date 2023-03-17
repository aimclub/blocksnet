from loguru import logger
import requests
import osm2geojson
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon
from functools import reduce
import osmnx as ox


class Blocks_model:

    """
    A class used to generate city blocks.

    ...

    Attributes
    ----------
    city_name : str
        the target city
    city_crs : int
        city's local crs system
    city_admin_level : int
        city's administrative level from OpenStreetMaps

    Methods
    -------
    get_blocks(self)
    """

    def __init__(self, city_name:str='Санкт-Петербург', city_crs:int=32636, city_admin_level:int=5):
        self.city_name = city_name # city name as a key in a query. City name is the same as city's name in OSM.
        self.city_crs = city_crs # city crs must be specified for more accurate spatial calculations.
        self.city_admin_level = city_admin_level # administrative level must be specified for overpass turbo query.
        self.global_crs = 4326 # globally used crs.
        self.ROAD_WIDTH = 5 # road geometry buffer. So road geometries won't be thin as a line.
        self.WATER_WIDTH = 1 # water geometry buffer. So water geometries in some cases won't be thin as a line.
        self.cutoff_ratio = 0.15 # polygon's perimeter to area ratio. Objects with bigger ration will be dropped.
        self.cutoff_area = 1400 # in meters. Objects with smaller area will be dropped.

    def _get_city_geometry(self) -> gpd.GeoDataFrame:

        """ 
        This function downloads the geometry bounds of the chosen city
        and concats it into one solid polygon (/multipolygon)

        
        Parameters
        ----------
        self 


        Returns
        -------
        city_geometry : GeoDataFrame
            Geometry of city.
        """

        logger.info('Getting city geometry')
    
        overpass_url = "http://overpass-api.de/api/interpreter"
        overpass_query = f"""
        [out:json];
                area['name'='{self.city_name}']->.searchArea;
                (
                way["admin_level"="{self.city_admin_level}"](area.searchArea);
                relation["admin_level"="{self.city_admin_level}"](area.searchArea);
                );
        out geom;
        """
        result = requests.get(overpass_url, params={'data': overpass_query}).json()
        city_geometry = osm2geojson.json2geojson(result)
        city_geometry = gpd.GeoDataFrame.from_features(city_geometry["features"]).set_crs(self.global_crs).to_crs(self.city_crs)
        city_geometry = city_geometry[["id", "geometry"]]

        # Check if all geometries are valid
        city_geometry = city_geometry[city_geometry.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])]
        city_geometry = city_geometry.dissolve()

        return city_geometry
        
    def _fill_holes_in_blocks(row: gpd.GeoSeries) -> Polygon:

        """ 
        This geometry will be cut later from city's geometry.
        The water entities will split blocks from each other. The water geometries are taken using overpass turbo.
        The tags used in the query are: "natural"="water", "waterway"~"river|stream|tidal_channel|canal".

        
        Parameters
        ----------
        self 


        Returns
        -------
        water_geometry : GeoDataFrame
            Geometry of water. The water geometries are also buffered a little so the division of city's geometry
            could be more noticable.
        """

        logger.info('Filling rings')

        newgeom = None

        if len(row["rings"]) > 0:
                to_fill = [Polygon(ring) for ring in row["rings"]]
                if len(to_fill) > 0:
                    newgeom = reduce(lambda geom1, geom2: geom1.union(geom2),[row["geometry"]] + to_fill)

        if newgeom:
            return newgeom
        else:
            return row["geometry"]
    
    def _get_water_bounds(self) -> gpd.GeoDataFrame:

        """ 
        This geometry will be cut later from city's geometry.
        The water entities will split blocks from each other. The water geometries are taken using overpass turbo.
        The tags used in the query are: "natural"="water", "waterway"~"river|stream|tidal_channel|canal".

        
        Parameters
        ----------
        self 


        Returns
        -------
        water_geometry : GeoDataFrame
            Geometry of water. The water geometries are also buffered a little so the division of city's geometry
            could be more noticable.
        """

        logger.info('Getting water geometry')

        # Get water polydons in the city
        overpass_url = "http://lz4.overpass-api.de/api/interpreter"
        overpass_query = f"""
        [out:json];
                area[name="{self.city_name}"]->.searchArea;
                (
                relation["natural"="water"](area.searchArea);
                way["natural"="water"](area.searchArea);
                relation["waterway"~"river|stream|tidal_channel|canal"](area.searchArea);
                way["waterway"~"river|stream|tidal_channel|canal"](area.searchArea);
                );
        out geom;
        """
        result = requests.get(overpass_url, params={'data': overpass_query}).json()

        # Parse osm response
        resp = osm2geojson.json2geojson(result)
        water_bounds = gpd.GeoDataFrame.from_features(resp['features']).set_crs(self.global_crs).to_crs(self.city_crs)

        # Check if all geometries are valid
        water_bounds.loc[water_bounds.geometry.geom_type.isin(['LineString','MultiLineString']),
                          'geometry'] = water_bounds.geometry.buffer(self.WATER_WIDTH)

        return water_bounds

    def _get_rails_bounds(self) -> gpd.GeoDataFrame:

        """ 
        This geometry will be cut later from city's geometry.
        The railways will split blocks from each other. The railways geometries are taken using overpass turbo.
        The tags used in the query are: "railway"~"rail|light_rail"

        
        Parameters
        ----------
        self 


        Returns
        -------
        rails_geometry : GeoDataFrame
            Geometry of railways.
        """

        logger.info('Getting rails geometry')

        overpass_url = "http://lz4.overpass-api.de/api/interpreter"
        overpass_query = f"""
        [out:json];
                area[name="{self.city_name}"]->.searchArea;
                (
                relation["railway"~"rail|light_rail"](area.searchArea);
                way["railway"~"rail|light_rail"](area.searchArea);
                );
        out geom;
        """
        result = requests.get(overpass_url, params={'data': overpass_query}).json()

        # Parse osm response
        resp = osm2geojson.json2geojson(result)
        rail_bounds = gpd.GeoDataFrame.from_features(resp['features']).set_crs(self.global_crs).to_crs(self.city_crs)

        # Check if all geometries are valid
        rail_bounds = rail_bounds[rail_bounds.geometry.geom_type != "Point"]
        rail_bounds.geometry = rail_bounds.buffer(self.ROAD_WIDTH)
        
        return rail_bounds
    
    def _get_roads_bounds(self, city_geometry: gpd.GeoDataFrame) -> gpd.GeoDataFrame:

        """ 
        This geometry will be cut later from city's geometry.
        The roads will split blocks from each other. The road geometries are taken using osmnx.

        
        Parameters
        ----------
        self 
        city_geometry : GeoDataFrame
            a GeoDataFrame with city bounds with cutter railways


        Returns
        -------
        roads_geometry : GeoDataFrame
            Geometry of roads buffered by 5 meters so it could be cut from city's geometry
            in the next step
        """

        logger.info('Getting roads geometry')

        # Get drive roads from osmnx lib in city bounds
        road_bounds = ox.graph_from_polygon(city_geometry.to_crs(self.global_crs).geometry.item(), network_type = 'drive')
        road_bounds = ox.utils_graph.graph_to_gdfs(road_bounds, nodes=False)
        road_bounds = road_bounds.reset_index(level=[0,1]).reset_index(drop=True).to_crs(self.city_crs)
        road_bounds = road_bounds[['u','v','geometry']]

        # Buffer roads ways to get their close to actual size.
        road_bounds["geometry"] = road_bounds["geometry"].buffer(self.ROAD_WIDTH)

        return road_bounds
    
    def _fill_deadends(self, city_geometry: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        
        """
        Some roads make a deadend in the block. To get rid of such deadends the blocks' polygons
        are buffered and then unbuffered back. This procedure makes possible to fill deadends.

        
        Parameters
        ----------
        self 
        city_geometry : GeoDataFrame
            a GeoDataFrame with city bounds with cutter railways and roads


        Returns
        -------
        city_geometry : GeoDataFrame
            Geometry of the city without road deadends
        """
        
        city_geometry.explode(ignore_index=True)
        city_geometry.geometry = city_geometry.geometry.map(lambda road: road.buffer((self.ROAD_WIDTH + 1)).buffer(-(self.ROAD_WIDTH + 1)))
        
        return city_geometry

    def _split_city_geometry(self, city_geometry: gpd.GeoDataFrame) -> gpd.GeoDataFrame:

        """
        Gets city geometry to split it by dividers like different kind of roads. The splitted parts are city blocks.
        However, not each resulted geometry is a valid city block. So inaccuracies of this division would be removed
        in the next step.


        Parameters
        ----------
        self 
        city_geometry : GeoDataFrame
            a GeoDataFrame with city bounds


        Returns
        -------
        blocks : GeoDataFrame
            city bounds splitted by railways, roads and water. Resulted polygons are city blocks
        """

        # Subtract railways from city's polygon
        rail = self._get_rails_bounds()
        city_geometry = gpd.overlay(city_geometry, rail, how='difference')
        del rail

        # Subtract roads from city's polygon
        roads_buffered = self._get_roads_bounds(city_geometry)
        city_geometry = gpd.overlay(city_geometry, roads_buffered, how='difference')
        del roads_buffered

        # Fill deadends
        city_geometry = self._fill_deadends(city_geometry)

        # Subtract water from city's polygon.
        # Water must be substracted after filling road deadends so small cutted water polygons
        # would be kept
        water = self._get_water_bounds()
        blocks = gpd.overlay(city_geometry, water, how='difference')
        del water, city_geometry

        blocks = blocks.explode(ignore_index=True).to_crs(self.gobal_crs)
        blocks['rings'] = blocks.interiors
        blocks.geometry = blocks.apply(lambda x: self._fill_holes_in_blocks(x), axis=1)

        blocks = blocks.reset_index()[['index', 'geometry']].rename(columns={'index':'id'})

        return blocks
    
    def _drop_invalid_geometries(self, blocks: gpd.GeoDataFrame) -> gpd.GeoDataFrame:

        """
        Gets and clears blocks's geometries, dropping unniccessary geometries.
        There two criterieas to dicide whether drop the geometry or not.
        First -- if the total area of the polygon is not too small (not less than self.cutoff_area)
        Second -- if the ratio of perimeter to total polygon area not more than self.cutoff_ratio. 
        The second criteria means that polygon is not too narrow and could be a valid block.


        Parameters
        ----------
        self 
        blocks : GeoDataFrame
            a GeoDataFrame with blocks and unneccessary geometries such as roundabouts,
            other small polygons or very narrow geometries which happened to exist after
            cutting roads and water from the city's polygon, but have no value for master-planning
            purposes


        Returns
        -------
        blocks : GeoDataFrame
            a GeoDataFrame with substracted blocks without unneccessary geometries
        """
        # Transfer blocks' geometries to cea crs to make valid calculations of their areas
        blocks = blocks.to_crs("+proj=cea +lat_0=35.68250088833567 +lon_0=139.7671 +units=m")
        blocks['area'] = blocks['geometry'].area  

        # First criteria check: total area
        blocks = blocks[blocks['area'] > self.cutoff_area]       
        
        # Second criteria check: perimetr / total area ratio
        blocks['length'] = blocks['geometry'].length
        blocks['ratio'] = blocks['length'] / blocks['area']

        # Drop polygons with an aspect ratio less than the threshold
        blocks = blocks[blocks['ratio'] < self.cutoff_ratio] 
        blocks.drop(columns=['area', 'length', 'ration'], inplace=True)

        return blocks.to_crs(self.global_crs)

    def get_blocks(self):

        """
        Main method. 
        
        This method gets city's boundaries from OSM. Then iteratively water, roads and railways are cutted
        from city's geometry. After splitting city into blocks invalid blocks with bad geometries are removed.
        
        For big city it takes about ~ 1-2 hours to split city's geometry into thousands of blocks.
        It takes so long because splitting one big geometry (like city) by another big geometry (like roads
        or waterways) is computationally expensive. Splitting city by roads and water entities are two the most
        time consuming processes in this method.
        """

        city_geometry = self._get_city_geometry()
        blocks = self._split_city_geometry(city_geometry=city_geometry)
        blocks = self._drop_invalid_geometries(blocks=blocks)

        return blocks
