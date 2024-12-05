import geopandas as gpd
from .base_method import BaseMethod

CONNECTIVITY_COLUMN = "connectivity"

PLOT_KWARGS = {
    "column": CONNECTIVITY_COLUMN,
    "legend": True,
    "cmap": "cool",
}


class WeightedConnectivity(BaseMethod):
    """
    Provides methods for block connectivity assessment taking into account population and services within urban blocks.

    Methods
    -------
    plot(gdf, linewidth=0.1, figsize=(10, 10))
        Plots connectivity data on a map.
    calculate()
        Calculates connectivity for all blocks in the city model.
    """

    @staticmethod
    def plot(gdf: gpd.GeoDataFrame, linewidth: float = 0.1, figsize: tuple[int, int] = (10, 10)):
        """
        Plots connectivity data for blocks on a map.

        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            GeoDataFrame containing block geometries and connectivity data.
        linewidth : float, optional
            Line width for plotting the geometries, by default 0.1.
        figsize : tuple of int, optional
            Size of the figure to plot, by default (10, 10).

        Returns
        -------
        None
        """
        gdf.plot(linewidth=linewidth, figsize=figsize, **PLOT_KWARGS).set_axis_off()

    def calculate(self) -> gpd.GeoDataFrame:
        """
        Calculates weighted connectivity for all blocks in the city model.

        Connectivity is determined by the median value of the accessibility matrix row for each block.

        Returns
        -------
        geopandas.GeoDataFrame
            GeoDataFrame containing blocks with calculated connectivity.
        """
        blocks_gdf = self.city_model.get_blocks_gdf()[["geometry"]]
        blocks_gdf[CONNECTIVITY_COLUMN] = self.city_model.accessibility_matrix.median(axis=1)
        return blocks_gdf
