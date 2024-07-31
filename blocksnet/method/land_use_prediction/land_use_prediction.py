import numpy as np
import geopandas as gpd
import pandas as pd
from tqdm import trange
from sklearn.metrics.pairwise import cosine_similarity

from ...models.land_use import LandUse
from ..base_method import BaseMethod

PREDICTION_COLUMN = "land_use"


class LandUsePrediction(BaseMethod):
    """
    A class used to predict land use based on various methods including cosine similarity.

    Methods
    -------
    plot(gdf: gpd.GeoDataFrame, linewidth=0.1, figsize=(10, 10)):
        Plots the GeoDataFrame with predicted land use.

    _get_land_uses_services():
        Retrieves land use service types for the city model.

    _get_blocks_gdf():
        Retrieves block geometries and their service capacities as a GeoDataFrame.

    _get_unique_per_landuse(landuse_items):
        Finds unique service tags for each land use.

    _intersects(set1, set2):
        Checks if two sets have any common elements.

    _predict_block_landuse_cosine_similarity(block_vector, landuse_vectors, return_prob=False):
        Predicts land use for a block using cosine similarity.

    _predict_block_landuse(codes_in_block, landuse_items, use_cos_similarity=True):
        Predicts the land use for a block.

    calculate(use_cos_similarity=True):
        Calculates the land use prediction for all blocks.
    """

    @staticmethod
    def plot(gdf: gpd.GeoDataFrame, linewidth: float = 0.1, figsize: tuple[int, int] = (10, 10)):
        """
        Plots the GeoDataFrame with predicted land use.

        Parameters
        ----------
        gdf : gpd.GeoDataFrame
            GeoDataFrame containing the geometries and predicted land use.
        linewidth : float, optional
            Size of the polygon border to plot, by default 0.1.
        figsize : tuple, optional
            Size of the figure to plot, by default (10, 10).
        """
        ax = gdf.plot(color="#ddd", linewidth=linewidth, figsize=figsize)
        gdf.plot(ax=ax, column=PREDICTION_COLUMN, linewidth=linewidth, legend=True)
        ax.set_axis_off()

    def _get_land_uses_services(self) -> dict[str, list[str]]:
        """
        Retrieves land use service types for the city model.

        Returns
        -------
        dict
            Dictionary with land use names as keys and lists of service types as values.
        """
        get_land_use_services = lambda lu: [st.name for st in self.city_model.get_land_use_service_types(lu)]
        return {lu.name: get_land_use_services(lu) for lu in LandUse}

    def _get_blocks_gdf(self) -> gpd.GeoDataFrame:
        """
        Retrieves block geometries and their service capacities as a GeoDataFrame.

        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame with block geometries and boolean values representing the presence of each service type.
        """
        blocks = self.city_model.get_blocks_gdf()
        blocks_profiles = blocks[["geometry"]].copy()
        for service_type in self.city_model.service_types:
            column = f"capacity_{service_type.name}"
            if column in blocks.columns:
                blocks_profiles[service_type.name] = blocks[column].apply(lambda x: x > 0)
            else:
                blocks_profiles[service_type.name] = False
        # blocks_profiles['block_id'] = blocks_profiles.index
        return blocks_profiles

    @staticmethod
    def _get_unique_per_landuse(landuse_items) -> dict[LandUse, set[str]]:
        """
        Finds unique service tags for each land use.

        Parameters
        ----------
        landuse_items : dict
            Dictionary containing service codes or service tags for each land use.

        Returns
        -------
        dict
            Dictionary with land use categories as keys and sets of unique service tags as values.
        """

        unique_per_landuse = {}

        landuse_categories = landuse_items.keys()

        for landuse_category in landuse_categories:
            unique_elements = set(landuse_items[landuse_category])
            for other_list in [landuse_items[x] if x != landuse_category else [] for x in landuse_items]:
                unique_elements -= set(other_list)
            unique_per_landuse[landuse_category] = unique_elements

        return unique_per_landuse

    @staticmethod
    def _intersects(set1, set2) -> bool:
        """
        Checks if two sets have any common elements.

        Parameters
        ----------
        set1 : np.array | list | set
            First collection of elements.
        set2 : np.array | list | set
            Second collection of elements.

        Returns
        -------
        bool
            True if there are common elements, False otherwise.
        """
        if set1 is not set:
            set1 = set(set1)
        if set2 is not set:
            set2 = set(set2)

        return len(set1.intersection(set2)) > 0

    @staticmethod
    def _predict_block_landuse_cosine_similarity(block_vector, landuse_vectors, return_prob=False) -> str | None:
        """
        Predicts land use for a block using cosine similarity.

        Parameters
        ----------
        block_vector : list | np.array
            Collection of booleans representing services present in a block.
        landuse_vectors : dict
            Dictionary containing vectors representing services present in each land use.
        return_prob : bool, optional
            If True, also returns the probability of the prediction, by default False.

        Returns
        -------
        str or None
            Predicted land use category or None if no prediction is made.
        """

        similarity_dict = {}

        landuse_categories = landuse_vectors.keys()
        for category in landuse_categories:
            similarity = cosine_similarity([block_vector], [landuse_vectors[category]])[0][0]
            similarity_dict[category] = similarity

        predicted_landuse = max(similarity_dict, key=similarity_dict.get)
        predicted_landuse_probability = max(similarity_dict.values())

        if predicted_landuse_probability == 0:
            return None

        if return_prob:
            return (predicted_landuse, predicted_landuse_probability)
        return predicted_landuse

    def _predict_block_landuse(self, codes_in_block, landuse_items, use_cos_similarity=True):
        """
        Predicts the land use for a block.

        Parameters
        ----------
        codes_in_block : list | np.array
            List of service tags or service codes present in a block.
        landuse_items : dict
            Dictionary containing service codes or service tags for each land use.
        use_cos_similarity : bool, optional
            Use cosine similarity to predict unpredicted land uses for blocks, by default True.

        Returns
        -------
        str or None
            Predicted land use category or None if no prediction is made.
        """

        if len(codes_in_block) == 0:
            return None

        potential_landuse_categories = []
        unique_per_landuse = self._get_unique_per_landuse(landuse_items)
        landuse_categories = landuse_items.keys()
        for landuse_category in landuse_categories:
            if self._intersects(codes_in_block, unique_per_landuse[landuse_category]):
                potential_landuse_categories.append(landuse_category)

        if len(potential_landuse_categories) == 1:
            return potential_landuse_categories[0]

        if len(potential_landuse_categories) > 1:
            landuse_categories = potential_landuse_categories
            vector_header = np.array([j for i in landuse_items.values() for j in i])
            landuse_vectors = {y: [x in landuse_items[y] for x in vector_header] for y in landuse_categories}
            block_vector = [x in codes_in_block for x in vector_header]
            return self._predict_block_landuse_cosine_similarity(block_vector, landuse_vectors, return_prob=False)

        elif use_cos_similarity:
            vector_header = np.array([j for i in landuse_items.values() for j in i])
            landuse_vectors = {y: [x in landuse_items[y] for x in vector_header] for y in landuse_categories}
            block_vector = [x in codes_in_block for x in vector_header]
            return self._predict_block_landuse_cosine_similarity(block_vector, landuse_vectors, return_prob=False)

        else:
            return

    def calculate(self, use_cos_similarity=True):
        """
        Calculates the land use prediction for all blocks.

        Parameters
        ----------
        use_cos_similarity : bool, optional
            Use cosine similarity to predict unpredicted land uses for blocks, by default True.

        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame containing the geometries and predicted land use for all blocks.
        """

        blocks_profiles = self._get_blocks_gdf()
        landuse_items = self._get_land_uses_services()

        vector_header = np.array(blocks_profiles.drop(["geometry"], axis=1).columns)

        landuse_predictions = []
        for i in trange(len(blocks_profiles)):
            items_in_block = vector_header[blocks_profiles[vector_header].iloc[i]]
            landuse_prediction = self._predict_block_landuse(items_in_block, landuse_items, use_cos_similarity)
            landuse_predictions.append(landuse_prediction)

        res = blocks_profiles[["geometry"]].copy()
        res[PREDICTION_COLUMN] = landuse_predictions
        res = gpd.GeoDataFrame(res, crs=blocks_profiles.crs)

        return res
