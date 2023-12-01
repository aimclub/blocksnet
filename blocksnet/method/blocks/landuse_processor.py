import pandas as pd
import geopandas as gpd
import numpy as np
import shapely


def get_attributes_from_intersection(df,df_with_attribute,attribute_column,df_id_column='block_id',min_intersection=0.2,projected_crs=3857):
    df = df.to_crs(projected_crs)
    df_with_attribute = df_with_attribute.to_crs(projected_crs)
    
    df = df.drop(df_id_column,axis=1).reset_index().rename(columns={'index':df_id_column})
    
    df_temp = gpd.overlay(df[[df_id_column, "geometry"]],df_with_attribute[[attribute_column, "geometry"]],how="intersection",keep_geom_type=False)
    df_temp["intersection_area"] = df_temp["geometry"].area
    df_temp = df_temp.groupby([df_id_column, attribute_column])["intersection_area"].sum().reset_index()

    df["area"] = df["geometry"].area
    df_temp = df_temp.merge(df[[df_id_column, "area"]], how="left")
    df_temp["intersection_area"] = df_temp["intersection_area"] / df_temp["area"]
    df_temp = df_temp[df_temp['intersection_area']>min_intersection]
    
    res = df_temp.groupby(df_id_column)[attribute_column].apply(list).astype(str)
    
    return res


def add_landuse(blocks,pzz,zone_attribute='zone',min_intersection=0.2):
    
    blocks_pzz = blocks.copy()
    
    blocks_pzz[zone_attribute] = get_attributes_from_intersection(
        blocks_pzz,pzz,zone_attribute,min_intersection=min_intersection)
            
    pzz_codes = {'Д':'business',
                 'Ж':'living',
                 'П':'industrial',
                 'К':'special',
                 'Р':'recreation',
                 'С':'agriculture',
                 'У':'street',
                 'И':'transport'}
    
    landuse_categories = pd.DataFrame()
    
    for code,category in pzz_codes.items():
        landuse_categories[category] = blocks_pzz['zone'].str.contains(code)
    
    landuse_categories['transport'] += landuse_categories['street']
    landuse_categories = landuse_categories.drop('street',axis=1)
    
    landuse_categories = landuse_categories*landuse_categories.columns
    landuse_categories = landuse_categories.replace('',np.nan)
    blocks_pzz['landuse'] = landuse_categories.apply(lambda x: ' | '.join(x.dropna()),axis=1)
        
    blocks_pzz['landuse'] = blocks_pzz['landuse'].replace({
        'business | living':'mixed_use',
        'business | industrial':'industrial',
        '':np.nan})
    
    mixed_use_exceptions = ['ТД1-1','ТД1-1_1','ТД1-1_2','ТД1-2','ТД1-2_1','ТД1-2_2','Т3Ж1','Т3Ж2']
    mixed_use_mask = blocks_pzz['zone'].map(lambda x: str_contains_one_of(str(x),mixed_use_exceptions))
    
    blocks_pzz.loc[mixed_use_mask,'landuse'] = 'mixed-use'
    
    return blocks_pzz
    
    
# def cut_nodev(blocks, nodev, min_block_width=50):
    
#     nodev_roads = nodev.query('CODE_VID_Z=="ТУ"').copy()
#     nodev_other = nodev.query('CODE_VID_Z!="ТУ"').copy()
    
#     nodev_other = filter_bottlenecks(nodev_other,projected_crs,min_block_width).reset_index(drop=True)
#     nodev_other = nodev_other[np.logical_not(nodev_other.is_empty)].explode(index_parts=False)
#     nodev = pd.concat([nodev_other,nodev_roads])
    
#     # substract nodev geometry from geometry of generated blocks 
#     blocks = gpd.overlay(blocks,nodev,how="difference",keep_geom_type=False)
#     blocks = blocks.explode(index_parts=False)
#     blocks = blocks[blocks.type=='Polygon']
#     blocks['geometry'] = blocks.make_valid()
#     blocks = reindex_blocks(blocks)

#     # filter geometry after substracting nodev
#     blocks = filter_bottlenecks(blocks, local_crs)
#     #self.blocks = reindex_blocks(self.blocks)
    
#     # add nodev geometry to blocks
#     blocks = pd.concat([blocks, nodev[["geometry", "CODE_ZONE_"]]])
#     blocks = blocks.rename(columns={'CODE_ZONE_':'zone'})
#     blocks = blocks.explode(index_parts=False)
#     blocks = blocks.drop_duplicates(subset="geometry")
#     blocks = reindex_blocks(blocks)
    
#     # add new attribute to blocks – whether blocks are nodev or not
#     blocks["nodev"] = blocks["CODE_ZONE_"].notna()
    
#     blocks = reindex_blocks(blocks)
    
def str_contains_one_of(s,list_of_strings):
    return any([option in s for option in list_of_strings])