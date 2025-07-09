import xgboost as xgb
import numpy as np
from .base_strategy import XGBoostBaseStrategy


class XGBoostRegressionStrategy(XGBoostBaseStrategy):
    def __init__(self, model_params: dict | None = None):
        super().__init__(xgb.XGBRegressor, model_params)
