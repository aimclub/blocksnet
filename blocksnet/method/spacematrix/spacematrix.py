import pandas as pd
import geopandas as gpd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from ..base_method import BaseMethod

SM_MORPHOTYPE_COLUMN = "spacematrix_morphotype"
SM_CLUSTER_COLUMN = "spacematrix_cluster"
KB_MORPHOTYPE_COLUMN = "strelka_morphotype"


class Spacematrix(BaseMethod):
    """
    Provides methods for calculating and visualizing urban block morphotypes based on the Spacematrix model.

    Attributes
    ----------
    number_of_clusters : int
        Number of clusters for KMeans clustering.
    random_state : int
        Random state for reproducibility of clustering results.

    Methods
    -------
    plot(gdf, linewidth=0.1, figsize=(10, 10))
        Plots Spacematrix morphotypes on a map.
    calculate() -> gpd.GeoDataFrame
        Calculates Spacematrix and Strelka morphotypes for urban blocks.
    """

    number_of_clusters: int = 11
    random_state: int = 10

    @staticmethod
    def plot(gdf, linewidth: float = 0.1, figsize: tuple[int, int] = (10, 10)):
        """
        Plots Spacematrix morphotypes for blocks on a map.

        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            GeoDataFrame containing block geometries and Spacematrix morphotype data.
        linewidth : float, optional
            Line width for plotting the geometries, by default 0.1.
        figsize : tuple of int, optional
            Size of the figure to plot, by default (10, 10).

        Returns
        -------
        None
        """
        gdf.plot(column=SM_MORPHOTYPE_COLUMN, legend=True, linewidth=linewidth, figsize=figsize).set_axis_off()

    @staticmethod
    def _get_strelka_morphotypes(blocks) -> gpd.GeoDataFrame:
        """
        Calculates Strelka morphotypes for the given blocks.

        Parameters
        ----------
        blocks : geopandas.GeoDataFrame
            GeoDataFrame containing block geometries and attributes.

        Returns
        -------
        geopandas.GeoDataFrame
            GeoDataFrame with Strelka morphotypes added.
        """
        blocks = blocks.copy()
        storeys = [blocks["l"].between(0, 3), blocks["l"].between(4, 8), (blocks["l"] >= 9)]
        labels = ["Малоэтажная застройка", "Среднеэтажная застройка", "Многоэтажная застройка"]
        blocks["morphotype"] = np.select(storeys, labels, default="Другое")

        mxis = [
            (blocks["morphotype"] == "Малоэтажная застройка") & (blocks["mxi"] < 0.05),
            (blocks["morphotype"] == "Среднеэтажная застройка") & (blocks["mxi"] < 0.2),
            (blocks["morphotype"] == "Многоэтажная застройка") & (blocks["mxi"] < 0.1),
        ]
        labels = ["Малоэтажная нежилая застройка", "Среднеэтажная нежилая застройка", "Многоэтажная нежилая застройка"]
        blocks["morphotype"] = np.select(mxis, labels, default=blocks["morphotype"])

        conds = [
            (blocks["morphotype"] == "Малоэтажная застройка") & ((blocks["fsi"] * 10) <= 1),
            (blocks["morphotype"] == "Малоэтажная застройка") & ((blocks["fsi"] * 10) > 1),
            (blocks["morphotype"] == "Среднеэтажная застройка") & ((blocks["fsi"] * 10) <= 8) & (blocks["mxi"] < 0.45),
            (blocks["morphotype"] == "Среднеэтажная застройка") & ((blocks["fsi"] * 10) > 8) & (blocks["mxi"] < 0.45),
            (blocks["morphotype"] == "Среднеэтажная застройка") & ((blocks["fsi"] * 10) > 15) & (blocks["mxi"] >= 0.6),
            (blocks["morphotype"] == "Многоэтажная застройка") & ((blocks["fsi"] * 10) <= 15),
            (blocks["morphotype"] == "Многоэтажная застройка") & ((blocks["fsi"] * 10) > 15),
        ]
        labels = [
            "Индивидуальная жилая застройка",
            "Малоэтажная модель застройки",
            "Среднеэтажная микрорайонная застройка",
            "Среднеэтажная квартальная застройка",
            "Центральная модель застройки",
            "Многоэтажная советская микрорайонная застройка",
            "Многоэтажная соверменная микрорайонная застройка",
        ]
        blocks["morphotype"] = np.select(conds, labels, default=blocks["morphotype"])

        return blocks

    @staticmethod
    def _name_spacematrix_morphotypes(cluster) -> str:
        """
        Assigns names to Spacematrix morphotypes based on cluster data.

        Parameters
        ----------
        cluster : pandas.Series
            Series containing cluster data for each block.

        Returns
        -------
        str
            The name of the Spacematrix morphotype.
        """
        ranges = [[0, 3, 6, 10, 17], [0, 1, 2], [0, 0.22, 0.55]]
        labels = [
            ["Малоэтажный", "Среднеэтажный", "Повышенной этажности", "Многоэтажный", "Высотный"],
            [" низкоплотный", "", " плотный"],
            [" нежилой", " смешанный", " жилой"],
        ]
        cluster_name = []
        for ind in range(len(cluster)):
            cluster_name.append(
                labels[ind][[i for i in range(len(ranges[ind])) if cluster.iloc[ind] >= ranges[ind][i]][-1]]
            )
        return "".join(cluster_name)

    def _get_spacematrix_morphotypes(self, blocks) -> gpd.GeoDataFrame:
        """
        Calculates Spacematrix morphotypes for the given blocks using KMeans clustering.

        Parameters
        ----------
        blocks : geopandas.GeoDataFrame
            GeoDataFrame containing block geometries and attributes.

        Returns
        -------
        geopandas.GeoDataFrame
            GeoDataFrame with Spacematrix morphotypes and clusters added.
        """
        x = blocks[["fsi", "l", "mxi"]].copy()
        scaler = StandardScaler()
        x_scaler = pd.DataFrame(scaler.fit_transform(x))
        kmeans = KMeans(n_clusters=self.number_of_clusters, random_state=self.random_state, n_init="auto").fit(x_scaler)
        x["cluster"] = kmeans.labels_
        blocks = blocks.join(x["cluster"])
        cluster_grouper = blocks.groupby(["cluster"]).median(numeric_only=True)
        named_clusters = cluster_grouper[["l", "fsi", "mxi"]].apply(self._name_spacematrix_morphotypes, axis=1)
        blocks = blocks.join(named_clusters.rename("morphotype"), on="cluster")

        return blocks

    def calculate(self) -> gpd.GeoDataFrame:
        """
        Calculates Spacematrix and Strelka morphotypes for urban blocks.

        Returns
        -------
        geopandas.GeoDataFrame
            GeoDataFrame containing blocks with calculated Spacematrix and Strelka morphotypes.
        """
        blocks = self.city_model.get_blocks_gdf()
        developed_blocks = blocks.loc[blocks.footprint_area > 0]  # or osr>=10
        spacematrix_blocks = self._get_spacematrix_morphotypes(developed_blocks)
        strelka_blocks = self._get_strelka_morphotypes(developed_blocks)
        blocks[SM_MORPHOTYPE_COLUMN] = spacematrix_blocks["morphotype"]
        blocks[SM_CLUSTER_COLUMN] = spacematrix_blocks["cluster"]
        blocks[KB_MORPHOTYPE_COLUMN] = strelka_blocks["morphotype"]
        return blocks[["geometry", "l", "fsi", "mxi", KB_MORPHOTYPE_COLUMN, SM_CLUSTER_COLUMN, SM_MORPHOTYPE_COLUMN]]
