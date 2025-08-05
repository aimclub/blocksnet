from .base_strategy import SKLearnVotingBaseStrategy
from blocksnet.machine_learning.strategy import ClassificationBase


class SKLearnRegressionStrategy(SKLearnVotingBaseStrategy, ClassificationBase):
    pass
