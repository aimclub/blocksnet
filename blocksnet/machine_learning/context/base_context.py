from abc import ABC
from ..strategy.base_strategy import BaseStrategy


class BaseContext(ABC):
    def __init__(self, strategy: BaseStrategy):
        self.strategy = strategy

    def set_strategy(self, strategy: BaseStrategy):
        self.strategy = strategy

    def run(self, *args, **kwargs):
        if not self.strategy:
            raise ValueError("Strategy is not set.")
