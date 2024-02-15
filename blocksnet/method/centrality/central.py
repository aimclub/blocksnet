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

    def count_points_in_polygon(self, polygon, points) -> int:
        """
        Count the number of points within a given polygon.

        Parameters:
        - polygon: A polygon geometry within which points are counted.
        - points: A GeoDataFrame of point geometries to be tested against the polygon.

        Returns:
        - An integer count of points within the polygon.
        """
        return points.within(polygon).sum()

    def _shannon_diversity_index(self, counts) -> float:
        """
        Calculate the Shannon diversity index given counts.

        Parameters:
        - counts: A pandas Series representing counts of different categories.

        Returns:
        - The Shannon diversity index as a float.
        """
        proportions = counts / counts.sum()
        proportions = proportions[proportions > 0]
        return -np.sum(proportions * np.log(proportions))

    def calculate_shannon_diversity(self, points, polygons) -> gpd.GeoDataFrame:
        """
        Calculate Shannon diversity index for each polygon based on point data.

        Parameters:
        - points: A GeoDataFrame of point geometries.
        - polygons: A GeoDataFrame of polygon geometries.

        Returns:
        - A GeoDataFrame of polygons with an additional 'shannon' column representing the Shannon diversity index.
        """
        joined_gdf = gpd.sjoin(points, polygons, how="inner", op="within")
        counts_per_polygon = joined_gdf.groupby(['index_right', 'city_service_type']).size().unstack(fill_value=0)
        shannon_indexes = counts_per_polygon.apply(self._shannon_diversity_index, axis=1)
        polygons['shannon'] = polygons.index.map(shannon_indexes)
        return polygons

    def connectivity(self) -> gpd.GeoDataFrame:
        """
        Calculate connectivity for the city model.

        Returns:
        - A GeoDataFrame representing connectivity metrics.
        """
        return Connectivity(city_model=self.city_model).calculate()

    def centrality(self, points) -> gpd.GeoDataFrame:
        """
        Calculate centrality metrics for polygons based on point data and connectivity.

        Parameters:
        - points: A GeoDataFrame of point geometries.

        Returns:
        - A GeoDataFrame of polygons with added centrality metrics.
        """
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