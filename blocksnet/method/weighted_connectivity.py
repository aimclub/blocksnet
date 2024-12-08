import geopandas as gpd
from .base_method import BaseMethod
import numpy as np
import pandas as pd

CONNECTIVITY_COLUMN = "weighted_connectivity"

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

        blocks_gdf = self.city_model.get_blocks_gdf()[["geometry", "population"]]
        # Group services by block and count the number of services in each block
        blocks_gdf["service_count"] = blocks_gdf.apply(lambda s: len(self.city_model[s.name].all_services), axis=1)
        # services = self.city_model.get_services_gdf()
        # service_counts = services.groupby("block_id").size()
        # blocks_gdf["service_count"] = blocks_gdf.index.map(service_counts).fillna(0).astype(int)

        # Get accessibility matrix and corresponding population and service counts
        mx = self.city_model.accessibility_matrix
        populations = blocks_gdf.loc[mx.index, "population"]
        service_counts = blocks_gdf.loc[mx.index, "service_count"]

        # Compute accessibility from houses to services
        house_to_service = np.outer(populations, service_counts) / mx.values
        house_to_service = np.where(mx.values != 0, house_to_service, 0)  # handle zeros
        # Compute accessibility from services to houses
        service_to_house = np.outer(service_counts, populations) / mx.values
        service_to_house = np.where(mx.values != 0, service_to_house, 0)

        # Calculate the weighted accessibility matrix as the average of both directions
        weighted_mx = pd.DataFrame((house_to_service + service_to_house) / 2, index=mx.index, columns=mx.columns)

        blocks_gdf[CONNECTIVITY_COLUMN] = weighted_mx.mean(axis=1)
        return blocks_gdf[["geometry", CONNECTIVITY_COLUMN]]
