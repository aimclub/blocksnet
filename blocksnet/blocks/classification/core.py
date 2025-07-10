import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from blocksnet.machine_learning.context.base_context import BaseContext
from blocksnet.machine_learning.strategy.base_strategy import BaseStrategy
from blocksnet.machine_learning.strategy.classification_base import ClassificationBase
from .schemas import BlockCategory, BlocksGeometriesSchema, BlocksCategoriesSchema

CATEGORIES_LIST = list(BlockCategory)


class BlocksClassifier(BaseContext):
    def _preprocess_x(self, blocks_gdf: gpd.GeoDataFrame) -> np.ndarray:
        blocks_gdf = BlocksGeometriesSchema(blocks_gdf)
        return blocks_gdf.drop(columns=["geometry"]).values

    def _preprocess_y(self, blocks_gdf: gpd.GeoDataFrame) -> np.ndarray:
        blocks_gdf = BlocksCategoriesSchema(blocks_gdf)
        blocks_gdf.category = blocks_gdf.category.apply(lambda c: CATEGORIES_LIST.index(c))
        return blocks_gdf.values

    def _postprocess_y(self, y: np.ndarray, index: list[int]) -> pd.DataFrame:
        blocks_df = pd.DataFrame(data=y, index=index, columns=BlocksCategoriesSchema.columns_())
        blocks_df.category = blocks_df.category.apply(lambda c: CATEGORIES_LIST[c])
        return blocks_df

    def train(self, blocks_gdf: gpd.GeoDataFrame, metric=accuracy_score, split_params: dict | None = None):
        x = self._preprocess_x(blocks_gdf)
        y = self._preprocess_y(blocks_gdf)

        split_params = split_params or {"train_size": 0.8, "stratify": y, "random_state": 42}
        x_train, x_test, y_train, y_test = train_test_split(x, y, **split_params)

        self.strategy.train(x_train, x_test, y_train, y_test)
        y_pred = self.strategy.predict(x_test)
        return metric(y_pred, y_test)

    def run(self, blocks_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        x = self._preprocess_x(blocks_gdf)
        y = self.strategy.predict(x)
        return self._postprocess_y(y, index=blocks_gdf.index)
