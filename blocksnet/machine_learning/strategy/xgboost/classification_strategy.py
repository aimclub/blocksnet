import xgboost as xgb
import numpy as np
from .base_strategy import XGBoostBaseStrategy
from ..classification_base import ClassificationBase


class XGBoostClassificationStrategy(ClassificationBase, XGBoostBaseStrategy):
    def __init__(self, model_params: dict | None = None):
        super().__init__(xgb.XGBClassifier, model_params)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(x)
