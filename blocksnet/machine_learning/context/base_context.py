from abc import ABC, abstractmethod
from ..strategy.base_strategy import BaseStrategy


class BaseContext(ABC):
    """BaseContext class.

    """
    def __init__(self, strategy: BaseStrategy):
        """Initialize the instance.

        Parameters
        ----------
        strategy : BaseStrategy
            Description.

        Returns
        -------
        None
            Description.

        """
        self._strategy = strategy

    @property
    def strategy(self):
        """Strategy.

        """
        if self._strategy is None:
            raise ValueError("Strategy is not set")
        return self._strategy

    @strategy.setter
    def strategy(self, strategy: BaseStrategy):
        """Strategy.

        Parameters
        ----------
        strategy : BaseStrategy
            Description.

        """
        if not isinstance(strategy, BaseStrategy):
            raise TypeError("Strategy must be BaseStrategy at least")
        self._strategy = strategy

    @classmethod
    @abstractmethod
    def default(cls) -> "BaseContext":
        """Default.

        Returns
        -------
        "BaseContext"
            Description.

        """
        pass

    @abstractmethod
    def train(self, *args, **kwargs):
        """Train.

        Parameters
        ----------
        *args : tuple
            Description.
        **kwargs : dict
            Description.

        """
        pass

    @abstractmethod
    def run(self, *args, **kwargs):
        """Run.

        Parameters
        ----------
        *args : tuple
            Description.
        **kwargs : dict
            Description.

        """
        pass
