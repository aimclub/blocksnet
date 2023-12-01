import pandas as pd
import geopandas as gpd
import numpy as np
import shapely
from shapely.ops import polygonize
from shapely import (
    Point,
    MultiPoint,
    Polygon,
    MultiPolygon,
    LineString,
    MultiLineString,
)

import pyproj
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_crs_info
from pyproj import CRS
import blocksnet.preprocessing.blocks_validator


def verbose_print(text, verbose=True):
    if verbose: print(text)
    
    
class BlocksGenerator:
    
    def __init__(self, territory, roads=None, railways=None, water=None, verbose=True):
        
        self.validator = DataValidator()
        self.validator.validate(territory, roads, railways, water)
        
        self.verbose = verbose    
        
        self.water = None
        if water is not None:
            territory = territory.overlay(water[water.geom_type != 'LineString'],how='difference') # cut water polygons from territory
            self.water = water.geometry if type(water) == gpd.GeoDataFrame else water
            self.water = self.water.map(lambda x: x.boundary if x.geom_type in ['Polygon','MultiPolygon'] else x).set_crs(4326)
        
        self.territory = territory.geometry
        
        self._set_local_crs()

        self.roads = roads.geometry if type(roads) == gpd.GeoDataFrame else roads
        self.railways = railways.geometry if type(railways) == gpd.GeoDataFrame else railways

        self.blocks = None
                
            
    def generate_blocks(self, min_block_width=None):
        
        verbose_print("GENERATING BLOCKS", self.verbose)
        
        # create a GeoDataFrame with barriers
        barriers = gpd.GeoDataFrame(geometry=pd.concat([self.roads, self.water, self.railways]),crs=4326).reset_index(drop=True)
        barriers = barriers.explode(index_parts=True).reset_index(drop=True).geometry
        
        # transform enclosed barriers to polygons 
        verbose_print("Setting up enclosures...", self.verbose)
        blocks = self._get_enclosures(barriers,self.territory)

        # fill everything within blocks' boundaries
        verbose_print("Filling holes...", self.verbose)
        blocks = self._fill_holes(blocks)

        # cleanup after filling holes
        verbose_print("Dropping overlapping blocks...", self.verbose)
        blocks = self._drop_overlapping_blocks(blocks)
        blocks = blocks.explode(index_parts=False).reset_index(drop=True)

        blocks = blocks.rename(columns={"index": "block_id"})

        # apply negative and positive buffers consecutively to remove small blocks
        # and divide them on bottlenecks
        if min_block_width is not None:
            verbose_print("Filtering bottlenecks and small blocks...", self.verbose)
            blocks = self._filter_bottlenecks(blocks, self.local_crs, min_block_width)
            blocks = self._reindex_blocks(blocks)

        # calculate blocks' area in local projected CRS
        verbose_print("Calculating blocks area...", self.verbose)
        blocks["area"] = blocks.to_crs(self.local_crs).area
        blocks = blocks[blocks["area"] > 1]

        # fix blocks' indices
        blocks = self._reindex_blocks(blocks)

        self.blocks = blocks
        verbose_print("Blocks generated.\n", self.verbose)
        
        
    def explore(self,column=None,cmap='Blues',attribute='blocks',tiles='CartoDB Positron'):
        
        if attribute=='blocks':
            m = self.blocks.explore(column=column,tiles=tiles)
        return m
    
    
    def _set_local_crs(self):
        
        try:
            coords = [list(set(x)) for x in self.territory.envelope.boundary.coords.xy]

            area_of_interest = AreaOfInterest(
                west_lon_degree=coords[0][0],
                east_lon_degree=coords[0][1],
                north_lat_degree=coords[1][0],
                south_lat_degree=coords[1][1],
            )

            utm_crs_list = query_crs_info(
                pj_types=pyproj.enums.PJType.PROJECTED_CRS,
                area_of_interest=area_of_interest,
            )
            
            self.local_crs = CRS.from_epsg(utm_crs_list[0].code)
            
        except:
            self.local_crs = CRS(3857)
    
    
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
    def _fill_holes(blocks):
        
        blocks["geometry"] = blocks["geometry"].boundary
        blocks = blocks.explode(index_parts=False)
        blocks["geometry"] = blocks["geometry"].map(
            lambda x: Polygon(x) if x.geom_type != 'Point' else np.nan)
        blocks = blocks.dropna(subset='geometry').reset_index(drop=True).to_crs(4326)
        return blocks
    
    
    @staticmethod
    def _drop_overlapping_blocks(blocks):
        
        blocks = blocks.reset_index(drop=True)
        
        overlaps = blocks["geometry"].sindex.query(blocks["geometry"], predicate="contains")
        overlaps_dict = {x:[] for x in overlaps[0]}
        for x, y in zip(overlaps[0], overlaps[1]):
            if x != y:
                overlaps_dict[x].append(y)
                
        overlapping_block_indeces = list({x for v in overlaps_dict.values() for x in v})
        blocks = blocks.drop(overlapping_block_indeces)
        blocks = blocks.reset_index(drop=True)
        return blocks


    @staticmethod
    def _filter_bottlenecks(blocks,local_crs,min_width=40):
        
        def _filter_bottlenecks_helper(poly,min_width=40):
            try: return poly.intersection(poly.buffer(-min_width/2).buffer(min_width/2,join_style=2))
            except: return poly
        
        blocks['geometry'] = blocks.to_crs(local_crs)['geometry'].map(
            lambda x: _filter_bottlenecks_helper(x,min_width)).set_crs(local_crs).to_crs(4326)
        blocks = blocks[blocks['geometry'] != Polygon()]
        blocks = blocks.explode(index_parts=False)
        blocks = blocks[blocks.type == 'Polygon']
        
        if 'area' in blocks.columns:
            blocks['area'] = blocks.to_crs(local_crs).area
            
        return blocks
          
            
    @staticmethod
    def _reindex_blocks(blocks):
        
        if 'block_id' in blocks.columns:
            blocks = blocks.drop('block_id',axis=1).reset_index().rename(columns={'index':'block_id'})
        return blocks