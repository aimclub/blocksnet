import numpy as np
import geopandas as gpd
import pandas as pd
from tqdm import trange
from ...models.land_use import LandUse
from ..base_method import BaseMethod
from sklearn.metrics.pairwise import cosine_similarity

PREDICTION_COLUMN = "land_use"


class LandUsePrediction(BaseMethod):
    @staticmethod
    def plot(gdf: gpd.GeoDataFrame, figsize=(10, 10)):
        ax = gdf.plot(color="#ddd", figsize=figsize)
        gdf.plot(ax=ax, column=PREDICTION_COLUMN, legend=True)
        ax.set_axis_off()

    def _get_land_uses_services(self):
        get_land_use_services = lambda lu: [st.name for st in self.city_model.get_land_use_service_types(lu)]
        return {lu.name: get_land_use_services(lu) for lu in LandUse}

    def _get_blocks_gdf(self):
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
    def _get_unique_per_landuse(landuse_items):
        # landuse_items : dict – dictionary containing service codes or service tags for each land use

        # returns dict

        unique_per_landuse = {}

        landuse_categories = landuse_items.keys()

        for landuse_category in landuse_categories:
            unique_elements = set(landuse_items[landuse_category])
            for other_list in [landuse_items[x] if x != landuse_category else [] for x in landuse_items]:
                unique_elements -= set(other_list)
            unique_per_landuse[landuse_category] = unique_elements

        return unique_per_landuse

    @staticmethod
    def _intersects(set1, set2):
        # set1 : np.array | list | set
        # set2 : np.array | list | set

        # returns bool

        if set1 is not set:
            set1 = set(set1)
        if set2 is not set:
            set2 = set(set2)

        return len(set1.intersection(set2)) > 0

    @staticmethod
    def _predict_block_landuse_cosine_similarity(block_vector, landuse_vectors, return_prob=False):
        # block_vector : list | np.array – collection of booleans representing services present in a block
        # landuse_vectors : dict – dictionary containing vectors representing services present in each land use

        # returns str or None

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
        # codes_in_block : list | np.array – list of service tags or service codes present in a block
        # landuse_items : dict – dictionary containing service codes or service tags for each land use
        # use_cos_similarity : bool – use cosine similarity to predict unpredicted land uses for blocks

        # returns str or None

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
        # block_profile : geopandas.GeoDataFrame – geodataframe with service categories as columns and boolean values representing presence or absence of said service categories in a block
        # landuse_items : dict – dictionary containing service codes or service tags for each land use
        # use_cos_similarity : bool – use cosine similarity to predict unpredicted land uses for blocks

        # returns geopandas.GeoDataFrame

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
