import geopandas as gpd

from sklearn.preprocessing import MinMaxScaler
from ..base_method import BaseMethod
from ..connectivity import Connectivity, CONNECTIVITY_COLUMN
from ..diversity import Diversity, DIVERSITY_COLUMN

CENTRALITY_COLUMN = "centrality"
DENSITY_COLUMN = "density"

PLOT_KWARGS = {"column": CENTRALITY_COLUMN, "legend": True, "cmap": "RdYlGn", "vmin": 0}


class Centrality(BaseMethod):
    """
    Provides methods to analyze the distribution and diversity of points of interest in urban areas,
    measure connectivity, and compute centrality indices based on various geometric and network properties.

    Methods
    -------
    plot(gdf, linewidth=0.1, figsize=(10, 10))
        Plots centrality data on a map.
    diversity : gpd.GeoDataFrame
        Calculates diversity for the city model.
    connectivity : gpd.GeoDataFrame
        Calculates connectivity for the city model.
    calculate(connectivity_weight=1, density_weight=1, diversity_weight=1) -> gpd.GeoDataFrame
        Calculates centrality metrics for polygons based on point data and connectivity.
    """

    @staticmethod
    def plot(gdf: gpd.GeoDataFrame, linewidth: float = 0.1, figsize: tuple[int, int] = (10, 10)):
        """
        Plots centrality data for blocks on a map.

        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            GeoDataFrame containing block geometries and centrality data.
        linewidth : float, optional
            Line width for plotting the geometries, by default 0.1.
        figsize : tuple of int, optional
            Size of the figure to plot, by default (10, 10).

        Returns
        -------
        None
        """
        ax = gdf.plot(color="#ddd", linewidth=linewidth, figsize=figsize)
        gdf.plot(ax=ax, linewidth=linewidth, **PLOT_KWARGS)
        ax.set_axis_off()

    @property
    def diversity(self) -> gpd.GeoDataFrame:
        """
        Calculates diversity for the city model.

        Returns
        -------
        geopandas.GeoDataFrame
            GeoDataFrame containing blocks with calculated diversity of services.
        """
        div = Diversity(city_model=self.city_model)
        return div.calculate()

    @property
    def connectivity(self) -> gpd.GeoDataFrame:

        """
        Calculate connectivity for the city model.

        Returns:
        - A GeoDataFrame representing connectivity metrics.
        """
        conn = Connectivity(city_model=self.city_model)
        return conn.calculate()

    def calculate(
        self, connectivity_weight: float = 1, density_weight: float = 1, diversity_weight: float = 1
    ) -> gpd.GeoDataFrame:
        """
        Calculates centrality metrics for polygons based on point data and connectivity.

        Parameters
        ----------
        connectivity_weight : float, optional
            Weight for the connectivity metric, by default 1.
        density_weight : float, optional
            Weight for the density metric, by default 1.
        diversity_weight : float, optional
            Weight for the diversity metric, by default 1.

        Returns
        -------
        geopandas.GeoDataFrame
            GeoDataFrame containing blocks with calculated centrality metrics.
        """
        # get blocks diversity and connectivity indices
        blocks = self.city_model.get_blocks_gdf(True)
        blocks[DIVERSITY_COLUMN] = self.diversity[DIVERSITY_COLUMN]
        blocks[CONNECTIVITY_COLUMN] = self.connectivity[CONNECTIVITY_COLUMN]

        # calculate density as amount of services in each block
        services = self.city_model.get_services_gdf()
        blocks[DENSITY_COLUMN] = services.groupby("block_id").size() / blocks["site_area"]

        # normalize indices and calculate centrality index
        scaler = MinMaxScaler()
        blocks_normalized = blocks.copy()
        blocks_normalized[CONNECTIVITY_COLUMN] = 1 / blocks_normalized[CONNECTIVITY_COLUMN]
        blocks_normalized[[CONNECTIVITY_COLUMN, DENSITY_COLUMN, DIVERSITY_COLUMN]] = scaler.fit_transform(
            blocks[[CONNECTIVITY_COLUMN, DENSITY_COLUMN, DIVERSITY_COLUMN]]
        )
        blocks[CENTRALITY_COLUMN] = (
            density_weight * blocks_normalized[DENSITY_COLUMN]
            + diversity_weight * blocks_normalized[DIVERSITY_COLUMN]
            + connectivity_weight * blocks_normalized[CONNECTIVITY_COLUMN]
        )

        blocks[CENTRALITY_COLUMN] = blocks[CENTRALITY_COLUMN] / (
            connectivity_weight + density_weight + diversity_weight
        )

        return blocks[["geometry", CONNECTIVITY_COLUMN, DENSITY_COLUMN, DIVERSITY_COLUMN, CENTRALITY_COLUMN]]
