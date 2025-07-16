from abc import ABC, abstractmethod
from ..strategy.base_strategy import BaseStrategy


class BaseContext(ABC):
    def __init__(self, strategy: BaseStrategy):
        self._strategy = strategy

    @property
    def strategy(self):
        if self._strategy is None:
            raise ValueError("Strategy is not set")
        return self._strategy

    @strategy.setter
    def strategy(self, strategy: BaseStrategy):
        if not isinstance(strategy, BaseStrategy):
            raise TypeError("Strategy must be BaseStrategy at least")
        self._strategy = strategy

    @classmethod
    @abstractmethod
    def default(cls) -> "BaseContext":
        pass

    @abstractmethod
    def train(self, *args, **kwargs):
        pass

    @abstractmethod
    def run(self, *args, **kwargs):
        pass
