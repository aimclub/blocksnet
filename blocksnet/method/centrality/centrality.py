import geopandas as gpd

from sklearn.preprocessing import MinMaxScaler
from ..base_method import BaseMethod
from ..connectivity import Connectivity, CONNECTIVITY_COLUMN
from ..diversity import Diversity, DIVERSITY_COLUMN

CENTRALITY_COLUMN = "centrality"
DENSITY_COLUMN = "density"

PLOT_KWARGS = {"column": CENTRALITY_COLUMN, "legend": True, "cmap": "coolwarm", "vmin": -1, "vmax": 1}


class Centrality(BaseMethod):
    """
    This class allows us to analyse the distribution and diversity of points of interest in urban areas,
    measure connectivity and compute centrality indices based on various geometric and network properties.
    """

    @staticmethod
    def plot(gdf: gpd.GeoDataFrame, figsize=(10, 10)):
        ax = gdf.plot(color="#ddd", figsize=(10, 10))
        gdf.plot(ax=ax, **PLOT_KWARGS)
        ax.set_axis_off()

    @property
    def diversity(self) -> gpd.GeoDataFrame:
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
        Calculate centrality metrics for polygons based on point data and connectivity.

        Parameters:
        - points: A GeoDataFrame of point geometries.

        Returns:
        - A GeoDataFrame of polygons with added centrality metrics.
        """
        # get blocks diversity and connectivity indices
        blocks = self.city_model.get_blocks_gdf()
        blocks[DIVERSITY_COLUMN] = self.diversity[DIVERSITY_COLUMN]
        blocks[CONNECTIVITY_COLUMN] = self.connectivity[CONNECTIVITY_COLUMN]

        # calculate density as amount of services in each block
        services = self.city_model.get_services_gdf()
        blocks[DENSITY_COLUMN] = services.groupby("block_id").size() / blocks["site_area"]

        # normalize indices and calculate centrality index
        scaler = MinMaxScaler()
        blocks_normalized = blocks.copy()
        blocks_normalized[[CONNECTIVITY_COLUMN, DENSITY_COLUMN, DIVERSITY_COLUMN]] = scaler.fit_transform(
            blocks[[CONNECTIVITY_COLUMN, DENSITY_COLUMN, DIVERSITY_COLUMN]]
        )
        blocks[CENTRALITY_COLUMN] = (
            density_weight * blocks_normalized[DENSITY_COLUMN]
            + diversity_weight * blocks_normalized[DIVERSITY_COLUMN]
            - connectivity_weight * blocks_normalized[CONNECTIVITY_COLUMN]
        )

        return blocks[["geometry", CONNECTIVITY_COLUMN, DENSITY_COLUMN, DIVERSITY_COLUMN, CENTRALITY_COLUMN]]
