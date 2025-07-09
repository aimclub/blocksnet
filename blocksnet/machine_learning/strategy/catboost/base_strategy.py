from ..base_strategy import BaseStrategy
from catboost import CatBoost


class CatBoostBaseStrategy(BaseStrategy):
    def __init__(self, model_cls: type[CatBoost], model_params: dict | None = None):
        super().__init__(model_cls, model_params or {})
        raise NotImplementedError("TODO: to be impemented in the future")
