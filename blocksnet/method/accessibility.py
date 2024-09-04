import geopandas as gpd
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt

from ..models import Block
from .base_method import BaseMethod

ACCESSIBILITY_TO_COLUMN = "accessibility_to"
ACCESSIBILITY_FROM_COLUMN = "accessibility_from"


class Accessibility(BaseMethod):
    """
    Provides methods for block accessibility assessment.

    Methods
    -------
    plot(gdf, vmax, figsize=(10, 10))
        Plots accessibility data on a map.
    calculate(block)
        Calculates accessibility to and from a given block.
    """

    @staticmethod
    def plot(gdf: gpd.GeoDataFrame, vmax: float = 60, linewidth: float = 0.1, figsize: tuple[int, int] = (10, 10)):
        """
        Plots accessibility data for blocks on a map.

        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            GeoDataFrame containing block geometries and accessibility data.
        vmax : float
            Limits the upper value of the legend, by default 60.
        linewidth : float
            Size of polygons' borders, by default 0.1.
        figsize : tuple of int, optional
            Size of the figure to plot, by default (10, 10).

        Returns
        -------
        None
        """
        fig = plt.figure(figsize=figsize)
        grid = GridSpec(1, 2)

        ax_to = fig.add_subplot(grid[0, 0])
        gdf.plot(
            ax=ax_to, column=ACCESSIBILITY_TO_COLUMN, cmap="Greens", linewidth=linewidth, vmax=vmax, legend=True
        ).set_axis_off()
        ax_to.set_title("To other blocks")
        ax_to.set_axis_off()

        ax_from = fig.add_subplot(grid[0, 1])
        gdf.plot(
            ax=ax_from, column=ACCESSIBILITY_FROM_COLUMN, cmap="Blues", linewidth=linewidth, vmax=vmax, legend=True
        ).set_axis_off()
        ax_from.set_title("From other blocks")
        ax_from.set_axis_off()

    def calculate(self, block: Block | int) -> gpd.GeoDataFrame:
        """
        Calculates accessibility to and from a given block.

        Parameters
        ----------
        block : Block or int
            The block for which accessibility is calculated. It can be an instance
            of Block or an integer representing the block ID.

        Returns
        -------
        geopandas.GeoDataFrame
            GeoDataFrame containing blocks with calculated accessibility to and from
            the specified block.
        """
        blocks_gdf = self.city_model.get_blocks_gdf(True)[["geometry"]]
        blocks_gdf[ACCESSIBILITY_TO_COLUMN] = self.city_model.adjacency_matrix.loc[block.id]
        blocks_gdf[ACCESSIBILITY_FROM_COLUMN] = self.city_model.adjacency_matrix[block.id]
        return blocks_gdf
