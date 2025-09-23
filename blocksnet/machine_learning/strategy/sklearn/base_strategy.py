from abc import ABC
import joblib
import os
from sklearn.base import BaseEstimator
from ..base_strategy import BaseStrategy

MODEL_FILENAME = "model.joblib"


class SKLearnBaseStrategy(BaseStrategy, ABC):
    def __init__(self, model_cls: type[BaseEstimator], model_params: dict | None):
        super().__init__(model_cls, model_params or {})

    def _save_model(self, path: str):
        joblib.dump(self.model, os.path.join(path, MODEL_FILENAME))

    def _load_model(self, path: str):
        self.model = joblib.load(os.path.join(path, MODEL_FILENAME))
