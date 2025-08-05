from abc import ABC
from ..base_strategy import SKLearnBaseStrategy
from blocksnet.machine_learning.strategy import BaseStrategy


class SKLearnEnsembleBaseStrategy(SKLearnBaseStrategy, ABC):
    def __init__(self, strategies: list[BaseStrategy], model_cls, model_params: dict | None = None):
        super().__init__(model_cls=model_cls, model_params=model_params)
        self.strategies = strategies
