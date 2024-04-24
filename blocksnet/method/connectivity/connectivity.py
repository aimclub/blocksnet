import statistics

import geopandas as gpd

from ..base_method import BaseMethod

CONNECTIVITY_COLUMN = "connectivity"


class Connectivity(BaseMethod):
    @staticmethod
    def plot(gdf: gpd.GeoDataFrame):
        gdf.plot(column=CONNECTIVITY_COLUMN, legend=True, cmap="cool").set_axis_off()

    def calculate(self):
        blocks_gdf = self.city_model.get_blocks_gdf()[["geometry"]]
        blocks_gdf[CONNECTIVITY_COLUMN] = self.city_model.adjacency_matrix.apply(lambda x: statistics.median(x), axis=1)
        return blocks_gdf
