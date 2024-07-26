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
    """
    Provides methods for assessing the diversity of services in city blocks.

    Methods
    -------
    plot(gdf, linewidth=0.1, figsize=(10, 10))
        Plots diversity data on a map.
    calculate()
        Calculates the diversity of services for each block in the city model.
    """

    @staticmethod
    def plot(gdf: gpd.GeoDataFrame, linewidth: float = 0.1, figsize: tuple[int, int] = (10, 10)):
        """
        Plots diversity data for blocks on a map.

        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            GeoDataFrame containing block geometries and diversity data.
        linewidth : float, optional
            Line width for plotting the geometries, by default 0.1.
        figsize : tuple of int, optional
            Size of the figure to plot, by default (10, 10).

        Returns
        -------
        None
        """
        ax = gdf.plot(color="#ddd", linewidth=linewidth, figsize=figsize)
        gdf.plot(**PLOT_KWARGS, linewidth=linewidth, ax=ax)
        ax.set_axis_off()
        ax.set_title("Services diversity")

    @staticmethod
    def _shannon_index(services_series) -> float:
        """
        Calculate the Shannon diversity index for given service counts.

        Parameters
        ----------
        services_series : pandas.Series
            Series representing counts of different service types in a block.

        Returns
        -------
        float
            The Shannon diversity index.
        """
        proportions = services_series / services_series.sum()
        proportions = proportions[proportions > 0]
        return -np.sum(proportions * np.log(proportions))

    def calculate(self) -> gpd.GeoDataFrame:
        """
        Calculates the diversity of services for each block in the city model.

        Diversity is determined using the Shannon diversity index based on the
        distribution of service types within each block.

        Returns
        -------
        geopandas.GeoDataFrame
            GeoDataFrame containing blocks with calculated diversity of services.
        """
        services = self.city_model.get_services_gdf()
        services_per_block = services.groupby(["block_id", "service_type"]).size().unstack(fill_value=0)
        blocks = self.city_model.get_blocks_gdf()[["geometry"]]
        blocks[DIVERSITY_COLUMN] = services_per_block.apply(self._shannon_index, axis=1)

        return blocks
