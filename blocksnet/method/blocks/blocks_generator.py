import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.ops import polygonize
import utils

#from blocksnet.preprocessing.blocks_validator import DataValidator
    
    
class BlocksGenerator:
    
    def __init__(self, territory, roads=None, railways=None, water=None, verbose=True):
        
        #self.validator = DataValidator()
        #self.validator.validate(territory, roads, railways, water)
        
        self.verbose = verbose    
        
        self.water = None
        if water is not None:
            territory = territory.overlay(water[water.geom_type != 'LineString'],how='difference') # cut water polygons from territory
            self.water = water.geometry if type(water) == gpd.GeoDataFrame else water
            self.water = self.water.map(lambda x: x.boundary if x.geom_type in ['Polygon','MultiPolygon'] else x).set_crs(4326)
        
        self.territory = territory.geometry
        
        self.local_crs = utils.get_projected_crs(self.territory.iloc[0])

        self.roads = roads.geometry if type(roads) == gpd.GeoDataFrame else roads
        self.railways = railways.geometry if type(railways) == gpd.GeoDataFrame else railways

        self.blocks = None
                
            
    def generate_blocks(self, min_block_width=None):
        
        utils.verbose_print("GENERATING BLOCKS", self.verbose)
        
        # create a GeoDataFrame with barriers
        barriers = gpd.GeoDataFrame(geometry=pd.concat([self.roads, self.water, self.railways]),crs=4326).reset_index(drop=True)
        barriers = barriers.explode(index_parts=True).reset_index(drop=True).geometry
        
        # transform enclosed barriers to polygons 
        utils.verbose_print("Setting up enclosures...", self.verbose)
        blocks = self._get_enclosures(barriers,self.territory)

        # fill everything within blocks' boundaries
        utils.verbose_print("Filling holes...", self.verbose)
        blocks = utils.fill_holes(blocks)

        # cleanup after filling holes
        utils.verbose_print("Dropping overlapping blocks...", self.verbose)
        blocks = utils.drop_contained_geometries(blocks)
        blocks = blocks.explode(index_parts=False).reset_index(drop=True)

        blocks = blocks.rename(columns={"index": "block_id"})

        # apply negative and positive buffers consecutively to remove small blocks
        # and divide them on bottlenecks
        if min_block_width is not None:
            utils.verbose_print("Filtering bottlenecks and small blocks...", self.verbose)
            blocks = utils.filter_bottlenecks(blocks, self.local_crs, min_block_width)
            blocks = self._reindex_blocks(blocks)

        # calculate blocks' area in local projected CRS
        utils.verbose_print("Calculating blocks area...", self.verbose)
        blocks["area"] = blocks.to_crs(self.local_crs).area
        blocks = blocks[blocks["area"] > 1]

        # fix blocks' indices
        blocks = self._reindex_blocks(blocks)

        self.blocks = blocks
        utils.verbose_print("Blocks generated.\n", self.verbose)
        
        
    def explore(self,column=None,cmap='Blues',attribute='blocks',tiles='CartoDB Positron'):
        
        if attribute=='blocks':
            m = self.blocks.explore(column=column,tiles=tiles)
        return m
    
    
    @staticmethod
    def _get_enclosures(barriers,limit):
        # limit should be a geodataframe or geoseries with with Polygon or MultiPolygon geometry
        
        barriers = pd.concat([barriers,limit.boundary]).reset_index(drop=True)

        unioned = barriers.unary_union
        polygons = polygonize(unioned)
        enclosures = gpd.GeoSeries(list(polygons), crs=barriers.crs)
        _, enclosure_idxs = enclosures.representative_point().sindex.query(limit.geometry, predicate="contains")
        enclosures = enclosures.iloc[np.unique(enclosure_idxs)]
        enclosures = enclosures.rename('geometry').reset_index()
        
        return enclosures
          
            
    @staticmethod
    def _reindex_blocks(blocks):
        
        if 'block_id' in blocks.columns:
            blocks = blocks.drop('block_id',axis=1).reset_index().rename(columns={'index':'block_id'})
        return blocks