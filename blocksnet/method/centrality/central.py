from typing import ClassVar

import geopandas as gpd
import osmnx as ox
import pandas as pd
import numpy as np

from scipy.stats import entropy
from sklearn.preprocessing import MinMaxScaler
from ...models import Block, ServiceType
from ..base_method import BaseMethod
from ..connectivity import Connectivity
 


class CentralPlace(BaseMethod):

    def count_points_in_polygon(self, polygon, points):
        return points.within(polygon).sum()


    def calculate_shannon_diversity(self, points, polygons):

        joined_gdf = gpd.sjoin(points, polygons, how="inner", op="within")

        counts_per_polygon = joined_gdf.groupby(['index_right', 'city_service_type']).size().unstack(fill_value=0)

        def shannon_diversity_index(counts):
            # Расчет доли каждого типа услуги
            proportions = counts / counts.sum()
            # Использование только ненулевых долей для избежания деления на ноль
            proportions = proportions[proportions > 0]
            # Расчет индекса Шеннона
            return -np.sum(proportions * np.log(proportions))

        # Применение функции расчета индекса к каждой строке
        shannon_indexes = counts_per_polygon.apply(shannon_diversity_index, axis=1)

        polygons['shannon'] = polygons.index.map(shannon_indexes)

        return polygons

        
    
    def connectivity(self) -> gpd.GeoDataFrame:
        return Connectivity(city_model=self.city_model).calculate()


    def centrality(self, points):
        polygons = self.connectivity()
        polygons['density'] = polygons['geometry'].apply(lambda x: self.count_points_in_polygon(x, points['geometry']))
        polygons = polygons.loc[polygons.density > 0]
        polygons = self.calculate_shannon_diversity(points, polygons)
        scaler = MinMaxScaler()
        df_normalized = polygons.copy()
        df_normalized[['median', 'density', 'shannon']] = scaler.fit_transform(polygons[['median', 'density', 'shannon']])

        weights = {'median': 1, 'density': 1, 'shannon': 2}
        polygons['centrality'] = (
            weights['density'] * df_normalized['density'] +
            weights['shannon'] * df_normalized['shannon'] -
            weights['median'] * df_normalized['median'])
        return polygons
        
        