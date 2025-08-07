import numpy as np
import pandas as pd
import geopandas as gpd
from typing import Optional, Tuple, List

from libpysal.weights import W
from shapely import make_valid
from sklearn.neighbors import NearestNeighbors

from blocksnet.enums import LandUse

ZONES = {
    'rec_spec_agri': [
        LandUse.RECREATION,
        LandUse.SPECIAL,
        LandUse.AGRICULTURE,
    ],
    'bus_res': [
        LandUse.RESIDENTIAL,
        LandUse.BUSINESS,
    ],
    'industrial': [
        LandUse.INDUSTRIAL,
    ],
    'transport': [
        LandUse.TRANSPORT,
    ],
}


class DataProcessor:
    def __init__(self, buffer_distance: float = 1000, k_neighbors: int = 5):
        """Инициализация обработчика данных с настройками расстояния буфера и количества соседей.
        
        Args:
            buffer_distance: Расстояние для буферизации при подсчете близлежащих зон
            k_neighbors: Количество соседей для KNN
        """
        self.buffer_distance = buffer_distance
        self.k_neighbors = k_neighbors
        self.knn_model = None
        
        # Списки фичей для контекста и логарифмирования
        self.feature_names_for_spatial_context = [
            'mbr_area', 'solidity', 'compactness', 'shape_index', 
            'mbr_aspect_ratio', 'squareness_index', 'fractal_dimension',
            'rectangularity_index',
        ] + [f'nearby_{k}_count' for k in ZONES.keys()]

        
        self.columns_to_log = [
            'shape_index', 'mbr_area', 'mbr_aspect_ratio', 
            'solidity', 'asymmetry_x', 'asymmetry_y'
        ]

    def fit_knn_weights(self, gdf: gpd.GeoDataFrame) -> Tuple[W, NearestNeighbors]:
        """Создает KNN веса для пространственного контекста.
        
        Args:
            gdf: GeoDataFrame с полигонами
            
        Returns:
            Кортеж (веса libpysal, обученная KNN модель)
        """
        centroids = gdf.geometry.centroid
        coords = np.column_stack((centroids.x, centroids.y))
        
        if coords.size == 0:
            dummy = np.zeros((1, 2))
            return W({}, {}), NearestNeighbors(n_neighbors=1).fit(dummy)
            
        actual_k = min(self.k_neighbors, len(coords))
            
        knn = NearestNeighbors(n_neighbors=actual_k, n_jobs=-1).fit(coords)
        distances, indices = knn.kneighbors(coords)
        
        neighbors = {i: list(indices[i]) for i in range(len(coords))}
        weights = {i: [1.0]*len(indices[i]) for i in range(len(coords))}
        
        return W(neighbors, weights), knn

    def calc_polygon_features(self, gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        """Вычисляет геометрические характеристики полигонов.
        
        Args:
            gdf: GeoDataFrame с полигонами
            
        Returns:
            DataFrame с вычисленными характеристиками
        """
        try:
            geo = gdf.geometry
            area = geo.area.to_numpy()
            length = geo.length.to_numpy()
            centroids = geo.centroid
            cx, cy = centroids.x.to_numpy(), centroids.y.to_numpy()
            
            min_env = geo.minimum_rotated_rectangle()
            mbr_area = min_env.area.to_numpy()
            convex = geo.convex_hull
            convex_area = convex.area.to_numpy()
            
            # Вычисление различных метрик формы
            compactness = np.where(length>0, 4*np.pi*area/length**2, 0)
            fractal_dim = np.where((area>0)&(length>0)&(np.log(length)!=0), np.log(area)/np.log(length), 0)
            rectangularity = np.where(mbr_area>0, area/mbr_area, 0)
            
            bounds = min_env.bounds
            dx = bounds.maxx - bounds.minx
            dy = bounds.maxy - bounds.miny
            
            aspect_ratio = np.where((dx>0)&(dy>0), np.maximum(dx,dy)/np.minimum(dx,dy), 0)
            squareness = np.where(np.maximum(dx,dy)>0, np.minimum(dx,dy)/np.maximum(dx,dy), 0)
            shape_index = np.where(length>0, area / length, 0)
            solidity = np.where(convex_area>0, area/convex_area, 0)
            
            asym_x = np.abs((bounds.minx+bounds.maxx)/2 - cx)
            asym_y = np.abs((bounds.miny+bounds.maxy)/2 - cy)
            
            result = pd.DataFrame({
                'compactness': compactness,
                'fractal_dimension': fractal_dim,
                'shape_index': shape_index,
                'mbr_area': mbr_area,
                'rectangularity_index': rectangularity,
                'mbr_aspect_ratio': aspect_ratio,
                'squareness_index': squareness,
                'solidity': solidity,
                'asymmetry_x': asym_x,
                'asymmetry_y': asym_y,
            }, index=gdf.index)
            return result
            
        except Exception as e:
            raise f"Ошибка при вычислении характеристик: {str(e)}"

    def count_nearby_zones(self, gdf: gpd.GeoDataFrame, rec_gdf: Optional[gpd.GeoDataFrame], 
                          buffer_distance: float) -> pd.Series:
        """Подсчитывает количество зон определенного типа в буфере вокруг каждого объекта.
        
        Args:
            gdf: Основной GeoDataFrame
            rec_gdf: GeoDataFrame с зонами для подсчета
            buffer_distance: Расстояние буфера
            
        Returns:
            Series с количеством зон для каждого объекта
        """

        if rec_gdf is None or rec_gdf.empty:
            return pd.Series(0, index=gdf.index)
            
        try:
            # Проверка и преобразование CRS
            if gdf.crs != rec_gdf.crs:
                rec = rec_gdf.to_crs(gdf.crs)
            else:
                rec = rec_gdf
                
            buffers = gdf.geometry.buffer(buffer_distance)
            buff_gdf = gpd.GeoDataFrame(geometry=buffers, crs=gdf.crs)
            
            joined = gpd.sjoin(buff_gdf, rec[['geometry']], how='left', predicate='intersects')
            counts = joined.groupby(joined.index).size()
            
            result = counts.reindex(gdf.index, fill_value=0).astype(int)
            return result
            
        except Exception as e:
            raise f"Ошибка при подсчете зон: {str(e)}"

    def neighbor_stats(self, gdf: gpd.GeoDataFrame, knn: NearestNeighbors, 
                      context_df: Optional[pd.DataFrame], feature_names: List[str]) -> pd.DataFrame:
        """Вычисляет статистики соседей для пространственного контекста.
        
        Args:
            gdf: GeoDataFrame с объектами
            knn: Обученная KNN модель
            context_df: DataFrame с данными для контекста
            feature_names: Список имен признаков
            
        Returns:
            DataFrame со средними значениями признаков соседей
        """        
        centroids = gdf.geometry.centroid
        coords = np.column_stack((centroids.x, centroids.y))
        
        if context_df is None or coords.size == 0:
            return pd.DataFrame(0.0, index=gdf.index, 
                              columns=[f'context_neighbor_mean_{f}' for f in feature_names])
        
        try:
            distances, indices = knn.kneighbors(coords)
            arr = context_df[feature_names].to_numpy()
            neighbor_means = np.nanmean(arr[indices, :], axis=1)
            
            result = pd.DataFrame(
                neighbor_means, 
                index=gdf.index, 
                columns=[f'context_neighbor_mean_{f}' for f in feature_names]
            )
            
            return result
            
        except Exception as e:
            raise f"Ошибка при вычислении статистик соседей: {str(e)}"

    def transform_features(self, gdf, known_gdf_for_rec_zones=None):
        """Трансформация признаков с обработкой невалидных геометрий"""
        
        gdf = gdf.copy()
        
        # Проверяем и исправляем невалидные геометрии
        gdf.geometry = gdf.geometry.apply(lambda geom: make_valid(geom) if not geom.is_valid else geom)
        
        centroids = gdf.geometry.centroid
        gdf['x_local'] = 0.0
        gdf['y_local'] = 0.0
        
        # Обработка координат относительно центра города
        if 'city' in gdf:
            
            def get_city_center(group):
                try:
                    # Исправляем геометрии перед объединением
                    valid_geoms = group.apply(lambda geom: make_valid(geom) if not geom.is_valid else geom)
                    union = valid_geoms.unary_union
                    if not union.is_valid:
                        union = make_valid(union)
                    return union.centroid
                except Exception as e:
                    # Возвращаем центр первого полигона в качестве fallback
                    return group.iloc[0].centroid
            
            cc_geom = gdf.groupby('city')['geometry'].apply(get_city_center)
            ccdf = cc_geom.apply(lambda p: pd.Series({'x': p.x, 'y': p.y}))
            
            gdf = gdf.join(ccdf, on='city')
            gdf['x_local'] = centroids.x - gdf['x']
            gdf['y_local'] = centroids.y - gdf['y']
            gdf = gdf.drop(columns=['x', 'y'])
        
        # Вычисление геометрических характеристик
        calc = self.calc_polygon_features(gdf)
        gdf = pd.concat([gdf, calc], axis=1)
        
        # Подсчет близлежащих зон разных типов
        for zone_key, land_uses in ZONES.items():
            if known_gdf_for_rec_zones is not None and 'category' in known_gdf_for_rec_zones:
                rec_z = known_gdf_for_rec_zones[known_gdf_for_rec_zones['category'].isin(land_uses)]
            else:
                rec_z = None

            gdf[f'nearby_{zone_key}_count'] = self.count_nearby_zones(gdf, rec_z, self.buffer_distance)

        
        return gdf

    def prepare_data(self, gdf: gpd.GeoDataFrame, is_train: bool = False, 
                   train_gdf_for_rec_zones: Optional[gpd.GeoDataFrame] = None, 
                   knn_model: Optional[NearestNeighbors] = None) -> pd.DataFrame:
        """Подготавливает данные для обучения/предсказания.
        
        Args:
            gdf: Входной GeoDataFrame
            is_train: Флаг обучения
            train_gdf_for_rec_zones: Данные для обучения (для теста)
            knn_model: Обученная KNN модель (для теста)
            
        Returns:
            DataFrame с подготовленными признаками
        """
        
        processed = self.transform_features(gdf, train_gdf_for_rec_zones)
        w, current_knn = self.fit_knn_weights(processed)
        
        if is_train:
            self.knn_model = current_knn
            context = self.neighbor_stats(processed, current_knn, processed, 
                                        self.feature_names_for_spatial_context)
        else:
            if knn_model is None:
                raise ValueError("You need to have a trained KNN model for prediction")
            context = self.neighbor_stats(processed, knn_model, train_gdf_for_rec_zones, 
                                        self.feature_names_for_spatial_context)
        
        processed = pd.concat([processed, context], axis=1)
        
        # Логарифмирование признаков
        for col in self.columns_to_log:
            if col in processed.columns:
                processed[f'{col}_log'] = np.log1p(processed[col])
        
        return processed