import networkx as nx
import pandas as pd
import numpy as np
from tqdm import tqdm
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from blocksnet.machine_learning import BaseContext
from blocksnet.enums import SettlementCategory
from blocksnet.config import log_config
from blocksnet.machine_learning.strategy import BaseStrategy, ClassificationBase
from .utils import CATEGORY_KEY, preprocess_graph, calculate_graph_features
from ._strategy import strategy

CATEGORIES_LIST = list(SettlementCategory)


class NetworkClassifier(BaseContext):
    def __init__(self, strategy: BaseStrategy):
        super().__init__(strategy=strategy)
        self._train_data: pd.DataFrame | None = None

    def prepare_train(self, graphs: list[nx.Graph]):
        self._train_data = self._get_features_df(graphs, with_category=True)

    @property
    def train_data(self) -> pd.DataFrame:
        if self._train_data is None:
            raise ValueError(f"No train data found. One must prepare it first using prepare_train() method.")
        return self._train_data.copy()

    def _get_features_df(self, graphs: list[nx.Graph], with_category: bool) -> pd.DataFrame:
        logger.info("Preprocessing graphs.")
        graphs = [
            preprocess_graph(g, validate_category=with_category) for g in tqdm(graphs, disable=log_config.disable_tqdm)
        ]
        features_dicts = []
        logger.info("Calculating graphs features.")
        for graph in tqdm(graphs, disable=log_config.disable_tqdm):
            features_dict = calculate_graph_features(graph)
            if with_category:
                category = graph.graph[CATEGORY_KEY]
                features_dict[CATEGORY_KEY] = CATEGORIES_LIST.index(category)
            features_dicts.append(features_dict)
        logger.success("Features are successfully built.")
        return pd.DataFrame(features_dicts)

    def _preprocess_x(self, features_df: pd.DataFrame) -> np.ndarray:
        if CATEGORY_KEY in features_df.columns:
            features_df = features_df.drop(columns=[CATEGORY_KEY])
        return features_df.values

    def _preprocess_y(self, features_df: pd.DataFrame) -> np.ndarray:
        return features_df[[CATEGORY_KEY]].values

    def train(self, metric=accuracy_score, split_params: dict | None = None):
        features_df = self.train_data

        x = self._preprocess_x(features_df)
        y = self._preprocess_y(features_df)

        split_params = split_params or {"train_size": 0.8, "stratify": y, "random_state": 42}
        x_train, x_test, y_train, y_test = train_test_split(x, y, **split_params)

        self.strategy.train(x_train, x_test, y_train, y_test)
        y_pred = self.strategy.predict(x_test)
        return metric(y_pred, y_test)

    def run(self, graphs: list[nx.Graph]) -> pd.DataFrame:
        features_df = self._get_features_df(graphs, with_category=False)

        x = self._preprocess_x(features_df)
        y_pred = self.strategy.predict(x)

        features_df[CATEGORY_KEY] = y_pred
        features_df[CATEGORY_KEY] = features_df[CATEGORY_KEY].apply(lambda c: CATEGORIES_LIST[c])

        if isinstance(self.strategy, ClassificationBase):
            y_proba = self.strategy.predict_proba(x)
            features_df[[c.value for c in CATEGORIES_LIST]] = y_proba

        return features_df

    @classmethod
    def default(cls) -> "NetworkClassifier":
        return cls(strategy)
