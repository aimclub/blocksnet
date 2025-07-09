import numpy as np
from catboost import CatBoostClassifier
from .base_strategy import CatBoostBaseStrategy
from ..classification_base import ClassificationBase


class CatBoostClassificationStrategy(ClassificationBase, CatBoostBaseStrategy):
    def __init__(self, model_params: dict | None = None):
        super().__init__(CatBoostClassifier, model_params)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        self._check_model()
        return self.model.predict_proba(x)
