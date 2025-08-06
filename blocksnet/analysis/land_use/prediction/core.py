import os
import logging
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from matplotlib import pyplot as plt
from shapely.geometry import mapping
from typing import Optional, Tuple, Dict, List, Union

from libpysal.weights import W

from shapely import make_valid
from sklearn.base import BaseEstimator
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split

from blocksnet.machine_learning.strategy.sklearn.ensemble.voting.classification_strategy import (
    SKLearnVotingClassificationStrategy
)
from .schemas import BlocksInputSchema

# Настройка логгера
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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
            'rectangularity_index', 'nearby_bus_res_count', 
            'nearby_transport_count', 'nearby_industrial_count',
            'nearby_rec_spec_agri_count'
        ]
        
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
            logger.warning("Пустые координаты, возвращаю dummy модель")
            dummy = np.zeros((1, 2))
            return W({}, {}), NearestNeighbors(n_neighbors=1).fit(dummy)
            
        actual_k = min(self.k_neighbors, len(coords))
        if actual_k < self.k_neighbors:
            logger.warning(f"Уменьшено k_neighbors с {self.k_neighbors} до {actual_k} из-за малого количества объектов")
            
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
            logger.error(f"Ошибка при вычислении характеристик: {str(e)}")
            raise

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
                logger.debug("Преобразование CRS для rec_gdf")
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
            logger.error(f"Ошибка при подсчете зон: {str(e)}")
            raise

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
            logger.warning("Нет данных для контекста, возвращаю нули")
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
            logger.error(f"Ошибка при вычислении статистик соседей: {str(e)}")
            raise

    def transform_features(self, gdf, known_gdf_for_rec_zones=None):
        """Трансформация признаков с обработкой невалидных геометрий"""
        logger.info("Начало трансформации признаков...")
        
        gdf = gdf.copy()
        
        # Проверяем и исправляем невалидные геометрии
        logger.info(f"Количество геометрий до проверки {gdf.shape[0]}")
        gdf.geometry = gdf.geometry.apply(lambda geom: make_valid(geom) if not geom.is_valid else geom)
        logger.info(f"Количество геометрий после проверки {gdf.shape[0]}")
        
        centroids = gdf.geometry.centroid
        gdf['x_local'] = 0.0
        gdf['y_local'] = 0.0
        
        # Обработка координат относительно центра города
        if 'city' in gdf:
            logger.debug("Вычисление локальных координат относительно центра города")
            
            def get_city_center(group):
                try:
                    # Исправляем геометрии перед объединением
                    valid_geoms = group.apply(lambda geom: make_valid(geom) if not geom.is_valid else geom)
                    union = valid_geoms.unary_union
                    if not union.is_valid:
                        union = make_valid(union)
                    return union.centroid
                except Exception as e:
                    logger.warning(f"Ошибка при вычислении центра города: {str(e)}")
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
        logger.info("Подсчет близлежащих зон разных типов...")
        for z in ['rec_spec_agri','bus_res','industrial','transport']:
            if known_gdf_for_rec_zones is not None and 'land_use' in known_gdf_for_rec_zones:
                rec_z = known_gdf_for_rec_zones[known_gdf_for_rec_zones['land_use']==z]
                logger.debug(f"Найдено {len(rec_z)} зон типа {z}")
            else:
                rec_z = None
                logger.debug(f"Нет данных для зон типа {z}")
                
            gdf[f'nearby_{z}_count'] = self.count_nearby_zones(gdf, rec_z, self.buffer_distance)
        
        logger.info("Трансформация признаков завершена")
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
                logger.error("Для тестовых данных нужна обученная KNN модель")
                raise ValueError("Для тестовых данных нужен обученный KNN-модель")
            context = self.neighbor_stats(processed, knn_model, train_gdf_for_rec_zones, 
                                        self.feature_names_for_spatial_context)
        
        processed = pd.concat([processed, context], axis=1)
        
        # Логарифмирование признаков
        for col in self.columns_to_log:
            if col in processed.columns:
                processed[f'{col}_log'] = np.log1p(processed[col])
        
        return processed


class SpatialClassifier(SKLearnVotingClassificationStrategy):
    def __init__(
        self,
        estimators: list[tuple[str, BaseEstimator]],
        model_params: dict | None = None,
        buffer_distance: float = 1000,
        k_neighbors: int = 5,
    ):
        """
        Классификатор с учетом пространственных характеристик.

        Args:
            estimators: список моделей (name, model)
            model_params: параметры для VotingClassifier
            buffer_distance: Расстояние для буферизации
            k_neighbors: Количество соседей для KNN
        """
        super().__init__(estimators=estimators, model_params=model_params)

        self.data_processor = DataProcessor(buffer_distance=buffer_distance, k_neighbors=k_neighbors)
        self.feature_cols = None
        self.target_col = None
        self.is_fitted = False
        self.class_names_ = None
        self.train_gdf_for_rec_zones = None
        self.processed_train_for_context = None

    def train(self, train_gdf: gpd.GeoDataFrame, target_col: str = 'land_use_code') -> None:
        """Обучение классификатора.
        
        Args:
            train_gdf: Данные для обучения
            target_col: Целевая переменная
        """
        train_gdf = BlocksInputSchema.validate(train_gdf)
        self.train_gdf_for_rec_zones = train_gdf.copy()
        
        # Подготовка данных
        logger.info("Подготовка данных для обучения")
        processed_train = self.data_processor.prepare_data(train_gdf, is_train=True)
        self.processed_train_for_context = processed_train
        
        # Обучение модели
        logger.info("Обучение модели")
        # Сохраняем признаки и целевую переменную
        excluded_columns = ['geometry', 'land_use', 'land_use_code', 'city', 'city_center']
        excluded_columns += self.data_processor.columns_to_log
        self.feature_cols = [c for c in processed_train.columns if c not in excluded_columns]
        self.target_col = target_col

        X = processed_train[self.feature_cols]
        y = processed_train[self.target_col]
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        super().train(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

        self.is_fitted = True
        self.class_names_ = self.model.classes_
        logger.info("Обучение классификатора завершено")

    def predict(self, new_gdf: gpd.GeoDataFrame) -> np.ndarray:
        """Предсказание на новых данных."""
        processed_new = self._prepare_for_prediction(new_gdf)
        X = processed_new[self.feature_cols]
        return super().predict(X)

    def predict_proba(self, new_gdf: gpd.GeoDataFrame) -> np.ndarray:
        """Вероятности классов для новых данных."""
        processed_new = self._prepare_for_prediction(new_gdf)
        X = processed_new[self.feature_cols]
        return super().predict_proba(X)

    def _prepare_for_prediction(self, new_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Подготовка данных для предсказания."""
        if not self.is_fitted:
            raise RuntimeError("Модель не обучена. Сначала вызовите .fit()")
            
        return self.data_processor.prepare_data(
            new_gdf,
            is_train=False,
            train_gdf_for_rec_zones=self.processed_train_for_context,
            knn_model=self.data_processor.knn_model
        )

    def save_train_data(self, filename: str) -> None:
        """Сохраняет обучающие данные в GeoJSON."""
        # Check if the model is fitted
        if not self.is_fitted:
            raise RuntimeError("Модель не обучена")
        self._save_geojson(self.train_gdf_for_rec_zones, filename)

    def save_test_data(self, test_gdf: gpd.GeoDataFrame, filename: str) -> None:
        """Сохраняет тестовые данные в GeoJSON."""
        self._save_geojson(test_gdf, filename)

    def save_mistakes(self, test_gdf: gpd.GeoDataFrame, 
                    predictions: np.ndarray,
                    filename: str) -> None:
        """Сохраняет ошибочные предсказания в GeoJSON."""
        test_data = test_gdf.copy()
        test_data['pred_class'] = predictions
        test_data['pred_name'] = [self.class_names_[c] for c in predictions]
        
        # Находим несовпадения
        mistakes = test_data[test_data['land_use_code'] != test_data['pred_class']]
        mistakes['mismatch'] = mistakes.apply(
            lambda row: f"{row['land_use']}->{row['pred_name']}", axis=1
        )
        
        self._save_geojson(mistakes, filename)

    def save_predictions_to_geojson(self, gdf: gpd.GeoDataFrame, 
                                predictions: np.ndarray,
                                probabilities: np.ndarray,
                                filename: str) -> None:
        """Упрощенная версия с GeoPandas"""
        result = gdf.copy()
        result['pred_class'] = predictions
        result['pred_name'] = [self.class_names_[c] for c in predictions]
        
        # Добавляем вероятности
        for i, cls in enumerate(self.class_names_):
            result[f'prob_{cls}'] = probabilities[:, i]
        
        self._save_geojson(result.round(4), filename)  # Округляем вероятности

    def _save_geojson(self, gdf: gpd.GeoDataFrame, filename: str) -> None:
        """Упрощенное сохранение GeoJSON через GeoPandas с обработкой множественных геометрий"""
        try:
            # Создаем папки если нужно
            filepath = Path(filename)
            os.makedirs(filepath.parent, exist_ok=True)
            
            # Создаем копию, чтобы не изменять исходный DataFrame
            save_gdf = gdf.copy()
            
            # Проверяем наличие нескольких геометрических столбцов
            geom_cols = [col for col in save_gdf.columns if save_gdf[col].dtype == 'geometry']
            
            if len(geom_cols) > 1:
                # Оставляем только основную геометрию, остальные конвертируем в WKT
                main_geom = geom_cols[0]  # Берем первую геометрию как основную
                for col in geom_cols[1:]:
                    save_gdf[col + '_wkt'] = save_gdf[col].to_wkt()
                    save_gdf = save_gdf.drop(columns=[col])
            
            # Конвертируем все столбцы в строковые для безопасности JSON
            for col in save_gdf.columns:
                if save_gdf[col].dtype == 'object' and col != save_gdf.geometry.name:
                    save_gdf[col] = save_gdf[col].astype(str)
            
            # Сохраняем через GeoPandas
            save_gdf.to_file(filename, driver='GeoJSON', encoding='utf-8')
            
            logger.info(f"Файл сохранен: {filename}")
        except Exception as e:
            logger.error(f"Ошибка сохранения {filename}: {str(e)}")
            raise


