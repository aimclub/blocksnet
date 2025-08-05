from abc import ABC
from ..base_strategy import BaseStrategy
from sklearn.ensemble import VotingClassifier


class SKLearnBaseStrategy(BaseStrategy, ABC):
    pass
