import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from blocksnet.machine_learning import BaseContext
from blocksnet.preprocessing.feature_engineering import generate_geometries_features
from .schemas import BlockCategory, BlocksSchema, BlocksCategoriesSchema
from ._strategy import strategy

CATEGORIES_LIST = list(BlockCategory)


class BlocksClassifier(BaseContext):
    """Supervised classifier predicting block categories.

    The classifier wraps a machine-learning strategy to preprocess block
    geometries, train a model and postprocess predictions.
    """
    def _preprocess_x(self, blocks_gdf: gpd.GeoDataFrame) -> np.ndarray:
        blocks_gdf = BlocksSchema(blocks_gdf)
        blocks_gdf = generate_geometries_features(blocks_gdf, radiuses=True, aspect_ratios=True, centerlines=True)
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
        """Fit the classifier on labelled block geometries.

        Parameters
        ----------
        blocks_gdf : geopandas.GeoDataFrame
            GeoDataFrame with geometries and block category labels.
        metric : callable, default=sklearn.metrics.accuracy_score
            Evaluation metric applied to the validation split.
        split_params : dict, optional
            Parameters forwarded to :func:`sklearn.model_selection.train_test_split`.

        Returns
        -------
        float
            Evaluation metric computed on the validation subset.
        """

        x = self._preprocess_x(blocks_gdf)
        y = self._preprocess_y(blocks_gdf)

        split_params = split_params or {"train_size": 0.8, "stratify": y, "random_state": 42}
        x_train, x_test, y_train, y_test = train_test_split(x, y, **split_params)

        self.strategy.train(x_train, x_test, y_train, y_test)
        y_pred = self.strategy.predict(x_test)
        return metric(y_pred, y_test)

    def run(self, blocks_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        """Predict block categories and probabilities for new geometries.

        Parameters
        ----------
        blocks_gdf : geopandas.GeoDataFrame
            GeoDataFrame containing geometries to classify.

        Returns
        -------
        pandas.DataFrame
            DataFrame with predicted categories and class probabilities.
        """

        x = self._preprocess_x(blocks_gdf)
        y = self.strategy.predict(x)
        y_df = self._postprocess_y(y, index=blocks_gdf.index)
        y_df[[bc.value for bc in CATEGORIES_LIST]] = self.strategy.predict_proba(x)
        return y_df

    @classmethod
    def default(cls) -> "BlocksClassifier":
        """Instantiate a classifier with the default training strategy.

        Returns
        -------
        BlocksClassifier
            Configured classifier ready for training or inference.
        """

        return cls(strategy)
