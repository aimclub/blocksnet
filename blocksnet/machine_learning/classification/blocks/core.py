from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from .common import ModelWrapper, BlockCategory
from .schemas import BlocksSchema, BlocksCategoriesSchema
from blocksnet.preprocessing.feature_engineering import generate_geometries_features

CATEGORY_COLUMN = "category"
PROBABILITY_COLUMN = "probability"

CURRENT_DIRECTORY = Path(__file__).parent
MODELS_DIRECTORY = CURRENT_DIRECTORY / "models"
MODEL_PATH = str(MODELS_DIRECTORY / "model.cbm")


class BlocksClassifier(ModelWrapper):
    def __init__(self, model_path: str = MODEL_PATH):
        ModelWrapper.__init__(self, model_path)

    def _initialize_x(self, blocks_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        gdf = BlocksSchema(blocks_gdf)
        gdf = generate_geometries_features(gdf, radiuses=False, aspect_ratios=True, centerlines=True, combinations=True)
        return gdf.drop(columns=["geometry"])

    def _initialize_y(self, blocks_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        df = BlocksCategoriesSchema(blocks_gdf)
        return df

    def get_train_data(self, blocks_gdf: gpd.GeoDataFrame, test: float, seed: int, *args, **kwargs):
        x = self._initialize_x(blocks_gdf)
        y = self._initialize_y(blocks_gdf)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test, random_state=seed, *args, **kwargs)
        return x_train, x_test, y_train, y_test

    def train(self):
        ...

    def test(self):
        ...

    def evaluate(self, blocks_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        x = self._initialize_x(blocks_gdf)
        columns = self.model.feature_names_
        out = self._evaluate_model(x[columns])
        df = pd.DataFrame(out, columns=[c.value for c in list(BlockCategory)], index=blocks_gdf.index)

        categories = df.idxmax(axis=1)
        probabilities = df.max(axis=1)
        df[CATEGORY_COLUMN] = categories.apply(lambda c: BlockCategory(c))
        df[PROBABILITY_COLUMN] = probabilities
        return df
