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
    """BlocksClassifier class.

    """
    def _preprocess_x(self, blocks_gdf: gpd.GeoDataFrame) -> np.ndarray:
        """Preprocess x.

        Parameters
        ----------
        blocks_gdf : gpd.GeoDataFrame
            Description.

        Returns
        -------
        np.ndarray
            Description.

        """
        blocks_gdf = BlocksSchema(blocks_gdf)
        blocks_gdf = generate_geometries_features(blocks_gdf, radiuses=True, aspect_ratios=True, centerlines=True)
        return blocks_gdf.drop(columns=["geometry"]).values

    def _preprocess_y(self, blocks_gdf: gpd.GeoDataFrame) -> np.ndarray:
        """Preprocess y.

        Parameters
        ----------
        blocks_gdf : gpd.GeoDataFrame
            Description.

        Returns
        -------
        np.ndarray
            Description.

        """
        blocks_gdf = BlocksCategoriesSchema(blocks_gdf)
        blocks_gdf.category = blocks_gdf.category.apply(lambda c: CATEGORIES_LIST.index(c))
        return blocks_gdf.values

    def _postprocess_y(self, y: np.ndarray, index: list[int]) -> pd.DataFrame:
        """Postprocess y.

        Parameters
        ----------
        y : np.ndarray
            Description.
        index : list[int]
            Description.

        Returns
        -------
        pd.DataFrame
            Description.

        """
        blocks_df = pd.DataFrame(data=y, index=index, columns=BlocksCategoriesSchema.columns_())
        blocks_df.category = blocks_df.category.apply(lambda c: CATEGORIES_LIST[c])
        return blocks_df

    def train(self, blocks_gdf: gpd.GeoDataFrame, metric=accuracy_score, split_params: dict | None = None):
        """Train.

        Parameters
        ----------
        blocks_gdf : gpd.GeoDataFrame
            Description.
        metric : Any, default: accuracy_score
            Description.
        split_params : dict | None, default: None
            Description.

        """
        x = self._preprocess_x(blocks_gdf)
        y = self._preprocess_y(blocks_gdf)

        split_params = split_params or {"train_size": 0.8, "stratify": y, "random_state": 42}
        x_train, x_test, y_train, y_test = train_test_split(x, y, **split_params)

        self.strategy.train(x_train, x_test, y_train, y_test)
        y_pred = self.strategy.predict(x_test)
        return metric(y_pred, y_test)

    def run(self, blocks_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        """Run.

        Parameters
        ----------
        blocks_gdf : gpd.GeoDataFrame
            Description.

        Returns
        -------
        pd.DataFrame
            Description.

        """
        x = self._preprocess_x(blocks_gdf)
        y = self.strategy.predict(x)
        y_df = self._postprocess_y(y, index=blocks_gdf.index)
        y_df[[bc.value for bc in CATEGORIES_LIST]] = self.strategy.predict_proba(x)
        return y_df

    @classmethod
    def default(cls) -> "BlocksClassifier":
        """Default.

        Returns
        -------
        "BlocksClassifier"
            Description.

        """
        return cls(strategy)
