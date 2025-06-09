import geopandas as gpd
from shapely.geometry import LineString, MultiLineString
from shapely.ops import unary_union
from blocksnet.machine_learning.classification import BlockCategory
import numpy as np

def merge_invalid_blocks(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    
    while True:
        valid_mask = gdf["category"].isin([BlockCategory.NORMAL, BlockCategory.LARGE])
        invalid_mask = gdf["category"] == BlockCategory.INVALID
        merge_candidates = gdf[invalid_mask]
        valid_blocks = gdf[valid_mask]

        if merge_candidates.empty:
            break

        # Создаем пространственный индекс для валидных блоков
        valid_sindex = valid_blocks.sindex
        
        merged_any = False
        drop_indices = []
        
        for idx, row in merge_candidates.iterrows():
            if idx not in gdf.index:
                continue
                
            target_geom = row.geometry
            # Находим потенциальных соседей через пространственный индекс
            possible_matches_idx = list(valid_sindex.query(target_geom, predicate="intersects"))
            possible_matches = valid_blocks.iloc[possible_matches_idx]
            
            if possible_matches.empty:
                continue
                
            # Векторизованный расчет длины пересечений
            def get_shared_length(intersection):
                if isinstance(intersection, LineString):
                    return intersection.length
                elif isinstance(intersection, MultiLineString):
                    return sum(line.length for line in intersection.geoms)
                return 0
                
            intersections = possible_matches.geometry.intersection(target_geom)
            shared_lengths = intersections.apply(get_shared_length)
            
            if shared_lengths.empty or shared_lengths.max() == 0:
                continue
                
            # Находим индекс блока с максимальной длиной пересечения
            matched_idx = possible_matches.index[shared_lengths.argmax()]
            
            # Объединяем геометрии
            new_geom = unary_union([target_geom, gdf.at[matched_idx, 'geometry']])
            gdf.at[matched_idx, 'geometry'] = new_geom
            drop_indices.append(idx)
            merged_any = True
            
            # Прерываем цикл после первого успешного слияния
            if merged_any:
                break
                
        if drop_indices:
            gdf = gdf.drop(index=drop_indices)
            
        if not merged_any:
            break

    return gdf.reset_index(drop=True)


def merge_empty_blocks(gdf: gpd.GeoDataFrame, buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    
    # Векторизованная проверка наличия зданий
    buildings_sindex = buildings.sindex
    gdf["has_buildings"] = gpd.GeoSeries(gdf.geometry).apply(
        lambda geom: len(buildings_sindex.query(geom, predicate="intersects")) > 0
    )

    while True:
        empty_mask = ~gdf["has_buildings"] & (gdf["category"] != BlockCategory.INVALID)
        empty_blocks = gdf[empty_mask]
        
        if empty_blocks.empty:
            break
            
        # Создаем пространственный индекс для пустых блоков
        empty_sindex = empty_blocks.sindex
        merged_any = False
        drop_indices = []
        
        for idx, row in empty_blocks.iterrows():
            if idx not in gdf.index:
                continue
                
            target_geom = row.geometry
            # Находим потенциальных соседей через пространственный индекс
            possible_matches_idx = list(empty_sindex.query(target_geom, predicate="intersects"))
            possible_matches = empty_blocks.iloc[possible_matches_idx]
            possible_matches = possible_matches[possible_matches.index != idx]
            
            if possible_matches.empty:
                continue
                
            # Векторизованный расчет длины пересечений
            def get_shared_length(intersection):
                if isinstance(intersection, LineString):
                    return intersection.length
                elif isinstance(intersection, MultiLineString):
                    return sum(line.length for line in intersection.geoms)
                return 0
                
            intersections = possible_matches.geometry.intersection(target_geom)
            shared_lengths = intersections.apply(get_shared_length)
            
            if shared_lengths.empty or shared_lengths.max() == 0:
                continue
                
            # Находим индекс блока с максимальной длиной пересечения
            matched_idx = possible_matches.index[shared_lengths.argmax()]
            
            # Объединяем геометрии
            new_geom = unary_union([target_geom, gdf.at[matched_idx, 'geometry']])
            gdf.at[matched_idx, 'geometry'] = new_geom
            drop_indices.append(idx)
            merged_any = True
            
            # Прерываем цикл после первого успешного слияния
            if merged_any:
                break
                
        if drop_indices:
            gdf = gdf.drop(index=drop_indices)
            
        if not merged_any:
            break

    return gdf.drop(columns="has_buildings").reset_index(drop=True)