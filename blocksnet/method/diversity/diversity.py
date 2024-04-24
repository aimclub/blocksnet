import numpy as np
import geopandas as gpd
from ..base_method import BaseMethod

DIVERSITY_COLUMN = "diversity"

PLOT_KWARGS = {
    "column": DIVERSITY_COLUMN,
    "cmap": "RdYlGn",
    "legend": True,
}


class Diversity(BaseMethod):
    @classmethod
    def plot(gdf: gpd.GeoDataFrame, figsize=(10, 10)):
        ax = gdf.plot(color="#ddd", figsize=figsize)
        gdf.plot(**PLOT_KWARGS, figsize=figsize, ax=ax)
        ax.set_axis_off()
        ax.set_title("Services diversity")

    @staticmethod
    def _shannon_index(services_series) -> float:
        """
        Calculate the Shannon diversity index given counts.

        Parameters:
        - counts: A pandas Series representing counts of different categories.

        Returns:
        - The Shannon diversity index as a float.
        """
        proportions = services_series / services_series.sum()
        proportions = proportions[proportions > 0]
        return -np.sum(proportions * np.log(proportions))

    def calculate(self):
        services = self.city_model.get_services_gdf()
        services_per_block = services.groupby(["block_id", "service_type"]).size().unstack(fill_value=0)
        blocks = self.city_model.get_blocks_gdf()[["geometry"]]
        blocks[DIVERSITY_COLUMN] = services_per_block.apply(self._shannon_index, axis=1)

        return blocks
