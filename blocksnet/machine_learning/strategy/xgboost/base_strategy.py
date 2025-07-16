import os
import numpy as np
import pandas as pd
import xgboost as xgb
from abc import ABC
from ..base_strategy import BaseStrategy

MODEL_FILENAME = "model.ubj"


class XGBoostBaseStrategy(BaseStrategy, ABC):
    def __init__(self, model_cls: type[xgb.XGBModel], model_params: dict | None = None):
        super().__init__(model_cls, model_params or {})

    def train(self, x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> float:
        self.model = self.model_cls(**self.model_params)
        self.model.fit(x_train, y_train, eval_set=[(x_test, y_test)], verbose=False)
        return self.model.score(x_test, y_test)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict(x)

    def _save_model(self, path: str):
        self.model.save_model(os.path.join(path, MODEL_FILENAME))

    def _load_model(self, path: str):
        model_path = os.path.join(path, MODEL_FILENAME)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        self.model = self.model_cls(**self.model_params)
        self.model.load_model(model_path)
