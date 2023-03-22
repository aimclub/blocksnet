from loguru import logger
import requests
import osm2geojson
import geopandas as gpd
from shapely.geometry import Polygon
from functools import reduce
import osmnx as ox
import pandas as pd


class Blocks_model:

    """
    A class used to generate city blocks.
    By default it is initialized for  Peterhof city in Saint Petersburg area with its local crs 32636

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

    def __init__(self, city_osm_id:int=337442, city_crs:int=32636, water_geometry=None,
                 roads_geometry:gpd.GeoDataFrame=None, railways_geometry:gpd.GeoDataFrame=None, 
                 nature_geometry_boundaries:gpd.GeoDataFrame=None, city_geometry:gpd.GeoDataFrame=None):

        '''
        TODO: поправить багу с вырезанием дорог т.к. сейчас виснет на них. Мб сменить версию геопандас.
        '''

        self.city_osm_id = city_osm_id
        '''city osm id as a key in a query. City name is the same as city's name in OSM.'''
        self.city_crs = city_crs 
        '''city crs must be specified for more accurate spatial calculations.'''
        self.global_crs = 4326 
        '''globally used crs.'''
        self.ROADS_WIDTH = self.RAILWAYS_WIDTH = self.NATURE_WIDTH = 5
        '''road geometry buffer in meters. So road geometries won't be thin as a line.'''
        self.WATER_WIDTH = 1 
        '''water geometry buffer in meters. So water geometries in some cases won't be thin as a line.'''
        self.unnecessary_geometry_cutoff_ratio = 0.15
        '''polygon's perimeter to area ratio. Objects with bigger ration will be dropped.'''
        self.unnecessary_geometry_cutoff_area = 1_400 
        '''in meters. Objects with smaller area will be dropped.'''
        self.park_size_cutoff_area = 10_000 
        '''in meters. Objects with smaller area will be dropped.'''
        self.water_geometry = water_geometry
        self.roads_geometry = roads_geometry
        self.railways_geometry = railways_geometry
        self.nature_geometry_boundaries = nature_geometry_boundaries
        self.city_geometry = city_geometry


    def _make_overpass_turbo_request(self, overpass_query, buffer_size:int=0):

        overpass_url = "http://overpass-api.de/api/interpreter"

        result = requests.get(overpass_url, params={'data': overpass_query}).json()

        # Parse osm response
        resp = osm2geojson.json2geojson(result)
        entity_geometry = gpd.GeoDataFrame.from_features(resp['features']).set_crs(self.global_crs).to_crs(self.city_crs)
        entity_geometry = entity_geometry[["id", "geometry"]]
        
        # Check if any geometry is Point type and drop it
        entity_geometry = entity_geometry.loc[~entity_geometry["geometry"].geom_type.isin(['Point'])]

        # Buffer geometry in case of line-kind objects like waterways, roads or railways
        if buffer_size:
            entity_geometry["geometry"] = entity_geometry["geometry"].buffer(buffer_size)

        return entity_geometry

    def _get_city_geometry(self) -> None:

        """ 
        This function downloads the geometry bounds of the chosen city
        and concats it into one solid polygon (or multipolygon)

        
        Returns
        -------
        city_geometry : GeoDataFrame
            Geometry of city. City geometry is not returned and setted as a class attribute.
        """
        if self.city_geometry:
            logger.info('Got uploaded city geometry')

        else:
            logger.info('City geometry are not provided. Getting city geometry from OSM via overpass turbo')

            overpass_query = f"""
                            [out:json];
                                (
                                    rel({self.city_osm_id});
                                );
                            out geom;
                            """

            self.city_geometry = self._make_overpass_turbo_request(overpass_query=overpass_query)

            logger.info('Got city geometry')
        
    def _fill_spaces_in_blocks(self, row: gpd.GeoSeries) -> Polygon:

        """ 
        This geometry will be cut later from city's geometry.
        The water entities will split blocks from each other. The water geometries are taken using overpass turbo.
        The tags used in the query are: "natural"="water", "waterway"~"river|stream|tidal_channel|canal".
        This function is designed to be used with 'apply' function, applied to the pd.DataFrame.

        
        Parameters
        ----------
        row : GeoSeries


        Returns
        -------
        water_geometry : Union[Polygon, Multipolygon]
            Geometry of water. The water geometries are also buffered a little so the division of city's geometry
            could be more noticable.
        """

        new_block_geometry = None

        if len(row["rings"]) > 0:
                empty_part_to_fill = [Polygon(ring) for ring in row["rings"]]

                if len(empty_part_to_fill) > 0:
                    new_block_geometry = reduce(lambda geom1, geom2: geom1.union(geom2),[row["geometry"]] + empty_part_to_fill)

        if new_block_geometry:
            return new_block_geometry
        
        else:
            return row["geometry"]
        
    def _get_water_geometry(self) -> None:

        """ 
        This geometry will be cut later from city's geometry.
        The water entities will split blocks from each other. The water geometries are taken using overpass turbo.
        The tags used in the query are: "natural"="water", "waterway"~"river|stream|tidal_channel|canal".


        Returns
        -------
        water_geometry : Union[Polygon, Multipolygon]
            Geometries of water. The water geometries are also buffered a little so the division of city's geometry
            could be more noticable. Water geometry is not returned and setted as a class attribute.
        """

        if self.water_geometry:
            logger.info('Got uploaded water geometries')

        else:
            logger.info('Water geometries are not provided. Getting water geometry from OSM via overpass turbo')

            # Get water polygons in the city

            overpass_query = f"""
                            [out:json];
                            rel({self.city_osm_id});map_to_area;
                            (
                                relation(area)["natural"="water"];
                                way(area)["natural"="water"];
                                relation(area)["waterway"~"river|stream|tidal_channel|canal"];
                                way(area)["waterway"~"river|stream|tidal_channel|canal"];
                            );
                            out geom;
                            """

            self.water_geometry = self._make_overpass_turbo_request(overpass_query=overpass_query, buffer_size=self.WATER_WIDTH)

            logger.info('Got water geometries')
            
    def _get_railways_geometry(self) -> None:

        """ 
        This geometry will be cut later from city's geometry.
        The railways will split blocks from each other. The railways geometries are taken using overpass turbo.
        The tags used in the query are: "railway"~"rail|light_rail"


        Returns
        -------
        railways_geometry : Union[Polygon, Multipolygon]
            Geometries of railways. Railways geometry is not returned and setted as a class attribute.
        """

        if self.railways_geometry:
            logger.info('Got uploaded railways geometries')

        else:
            logger.info('Railways geometries are not provided. Getting railways geometries from OSM via overpass turbo')

            overpass_query = f"""
                            [out:json];
                            rel({self.city_osm_id});map_to_area;
                            (
                                relation(area)["railway"~"rail|light_rail"];
                                way(area)["railway"~"rail|light_rail"];
                            );
                            out geom;
                            """

            self.railways_geometry = self._make_overpass_turbo_request(overpass_query=overpass_query, buffer_size=self.RAILWAYS_WIDTH)

            logger.info('Got railways geometries')
    
    def _get_roads_geometry(self) -> None:

        """ 
        This geometry will be cut later from city's geometry.
        The roads will split blocks from each other. The road geometries are taken using osmnx.


        Returns
        -------
        self.roads_geometry : Union[Polygon, Multipolygon]
            Geometries of roads buffered by 5 meters so it could be cut from city's geometry
            in the next step. Roads geometry is not returned and setted as a class attribute.
        """

        if self.roads_geometry:
            logger.info('Got uploaded roads geometries')

        else:
            logger.info('Roads geometries are not provided. Getting roads geometries from OSM via OSMNX')

            # Get drive roads from osmnx lib in city bounds
            self.roads_geometry = ox.graph_from_polygon(self.city_geometry.to_crs(self.global_crs).geometry.item(), network_type='drive')
            self.roads_geometry = ox.utils_graph.graph_to_gdfs(self.roads_geometry, nodes=False)
            self.roads_geometry = self.roads_geometry.reset_index(level=[0,1]).reset_index(drop=True).to_crs(self.city_crs)
            self.roads_geometry = self.roads_geometry[["geometry"]]

            # Buffer roads ways to get their close to actual size.
            self.roads_geometry["geometry"] = self.roads_geometry["geometry"].buffer(self.ROADS_WIDTH)

            logger.info('Got roads geometries')

    def _get_nature_geometry(self) -> None:

        """ 
        This geometry will be cut later from city's geometry.
        Big parks, cemetery and nature_reserve will split blocks from each other. Their geometries are taken using overpass turbo.


        Returns
        -------
        self.nature_geometry : Union[Polygon, Multipolygon]
            Geometries of nature entities buffered by 5 meters so it could be cut from city's geometry
            in the next step. Nature geometry is not returned and setted as a class attribute.
        """

        if self.nature_geometry_boundaries:
            logger.info('Got uploaded nature geometries')

        else:
            logger.info('Nature geometries are not provided. Getting nature geometries from OSM via overpass turbo')

            overpass_query_parks = f"""
                            [out:json];
                            rel({self.city_osm_id});map_to_area;
                            (
                                relation(area)["leisure"="park"];
                            );
                            out geom;
                            """
            
            overpass_query_greeners = f"""
                            [out:json];
                            rel({self.city_osm_id});map_to_area;
                            (
                                relation(area)["landuse"="cemetery"];
                                relation(area)["leisure"="nature_reserve"];
                            );
                            out geom;
                            """

            self.nature_geometry_boundaries = self._make_overpass_turbo_request(overpass_query=overpass_query_parks)
            self.nature_geometry_boundaries = self.nature_geometry_boundaries[self.nature_geometry_boundaries.area > self.park_size_cutoff_area]
            other_greeners_tmp = self._make_overpass_turbo_request(overpass_query=overpass_query_greeners)
            self.nature_geometry_boundaries = pd.concat([self.nature_geometry_boundaries , other_greeners_tmp])
            del other_greeners_tmp

            self.nature_geometry_boundaries["geometry"] = self.nature_geometry_boundaries.boundary.buffer(self.NATURE_WIDTH)
            # self.nature_geometry_boundaries.rename(columns={0:"geometry"}, inplace=True)

            logger.info('Got nature geometries')

    
    def _fill_deadends(self) -> None:
        
        """
        Some roads make a deadend in the block. To get rid of such deadends the blocks' polygons
        are buffered and then unbuffered back. This procedure makes possible to fill deadends.

        
        Returns
        -------
        self.city_geometry : GeoDataFrame
            Geometry of the city without road deadends. City geometry is not returned and setted as a class attribute.
        """
        self.city_geometry = self.city_geometry.explode(ignore_index=True) # To make multi-part geometries into several single-part so they coud be processed separatedly
        self.city_geometry["geometry"] = self.city_geometry["geometry"].map(lambda block: block.buffer(self.ROADS_WIDTH + 1).buffer(-(self.ROADS_WIDTH + 1)))

    def _cut_nature_geometry(self):
        # Subtract parks, cemetery and other nature from city's polygon.
        # This nature entities are substracted only by their boundaries since the could divide polygons into important parts for master-planning
        self._get_nature_geometry()
        logger.info('Cutting nature geometries')
        self.city_geometry = gpd.overlay(self.city_geometry, self.nature_geometry_boundaries, how='difference')
        del self.nature_geometry_boundaries

    def _cut_roads_geometry(self):
        # Subtract roads from city's polygon
        self._get_roads_geometry()
        logger.info('Cutting roads geometries')
        self.city_geometry = gpd.overlay(self.city_geometry, self.roads_geometry, how='difference')
        del self.roads_geometry
        logger.info('Cut and deleted roads geometries')

    def _cut_railways_geometry(self):
        # Subtract railways from city's polygon
        self._get_railways_geometry()
        logger.info('Cutting railways geometries')
        self.city_geometry = gpd.overlay(self.city_geometry, self.railways_geometry, how='difference')
        del self.railways_geometry
        logger.info('Cut and deleted railways geometries')

    def _cut_water_geometry(self):
        # Subtract water from city's polygon.
        # Water must be substracted after filling road deadends so small cutted water polygons would be kept
        self._get_water_geometry()
        logger.info('Cutting water geometries')
        self.city_geometry = gpd.overlay(self.city_geometry, self.water_geometry, how='difference')
        del self.water_geometry, self.city_geometry
        logger.info('Cut and deleted water geometries')
        logger.info('Deleted city geometry; Got blocks geometries')
        logger.info('Clearing blocks geometries; Filling spaces (rings) in blocks')

    def _process_blocks(self):
        self.city_geometry = self.city_geometry.explode(ignore_index=True)
        self.city_geometry['rings'] = self.city_geometry.interiors
        self.city_geometry["geometry"] = self.city_geometry[["geometry", "rings"]].apply(lambda row: self._fill_spaces_in_blocks(row), axis='columns')

        logger.info('Clearing overlaying geometries')
        # Drop overlayed geometries
        new_geometries = self.city_geometry.unary_union
        new_geometries = gpd.GeoDataFrame(new_geometries, geometry=0)
        self.city_geometry["geometry"] = new_geometries[0]
        del new_geometries

        self.city_geometry = self.city_geometry.reset_index()[['index', 'geometry']].rename(columns={'index':'id'})
        self.city_geometry = self.city_geometry.explode(ignore_index=True)

    def _split_city_geometry(self) -> gpd.GeoDataFrame:

        """
        Gets city geometry to split it by dividers like different kind of roads. The splitted parts are city blocks.
        However, not each resulted geometry is a valid city block. So inaccuracies of this division would be removed
        in the next step.

        Returns
        -------
        blocks : Union[Polygon, Multipolygon]
            city bounds splitted by railways, roads and water. Resulted polygons are city blocks
        """

        self._cut_roads_geometry()
        self._cut_railways_geometry
        self._cut_nature_geometry()
        self._fill_deadends()
        self._cut_water_geometry()
        self._process_blocks()       

        return self.city_geometry
    
    def _drop_unnecessary_geometries(self) -> None:

        """
        Gets and clears blocks's geometries, dropping unnecessary geometries.
        There two criterieas to dicide whether drop the geometry or not.
        First -- if the total area of the polygon is not too small (not less than self.cutoff_area)
        Second -- if the ratio of perimeter to total polygon area not more than self.cutoff_ratio. 
        The second criteria means that polygon is not too narrow and could be a valid block.

        A GeoDataFrame with blocks and unnecessary geometries such as roundabouts,
        other small polygons or very narrow geometries which happened to exist after
        cutting roads and water from the city's polygon, but have no value for master-planning
        purposes


        Returns
        -------
        blocks : GeoDataFrame
            a GeoDataFrame with substracted blocks without unnecessary geometries.
            Blocks geometry is not returned and setted as a class attribute.
        """

        logger.info('Dropping unnecessary geometries')

        self.city_geometry['area'] = self.city_geometry['geometry'].area  

        # First criteria check: total area
        self.city_geometry = self.city_geometry[self.city_geometry['area'] > self.unnecessary_geometry_cutoff_area]       
        
        # Second criteria check: perimetr / total area ratio
        self.city_geometry['length'] = self.city_geometry['geometry'].length
        self.city_geometry['ratio'] = self.city_geometry['length'] / self.city_geometry['area']

        # Drop polygons with an aspect ratio less than the threshold
        self.city_geometry = self.city_geometry[self.city_geometry['ratio'] < self.unnecessary_geometry_cutoff_ratio] 
        self.city_geometry = self.city_geometry.loc[:, ['id','geometry']]

    def get_blocks(self):

        """
        Main method. 
        
        This method gets city's boundaries from OSM. Then iteratively water, roads and railways are cutted
        from city's geometry. After splitting city into blocks invalid blocks with bad geometries are removed.
        
        For big city it takes about ~ 1-2 hours to split big city's geometry into thousands of blocks.
        It takes so long because splitting one big geometry (like city) by another big geometry (like roads
        or waterways) is computationally expensive. Splitting city by roads and water entities are two the most
        time consuming processes in this method.

        For cities over the Russia it could be better to upload water geometries from elsewhere, than taking
        them from OSM. Somehow they are poorly documented on OSM.

        
        Returns
        -------
        blocks
            a GeoDataFrame of city blocks in setted local crs (32636 by default)
        """

        self._get_city_geometry()
        self._split_city_geometry()
        self._drop_unnecessary_geometries()

        logger.info('Saving output GeoDataFrame with blocks')


        return self.city_geometry
