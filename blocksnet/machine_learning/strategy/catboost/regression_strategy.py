from catboost import CatBoostRegressor
from .base_strategy import CatBoostBaseStrategy


class CatBoostRegressionStrategy(CatBoostBaseStrategy):
    def __init__(self, model_params: dict | None = None):
        super().__init__(CatBoostRegressor, model_params)
