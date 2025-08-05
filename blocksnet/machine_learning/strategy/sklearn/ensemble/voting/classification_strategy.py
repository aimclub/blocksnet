from sklearn.ensemble import VotingClassifier
from .base_strategy import SKLearnVotingBaseStrategy
from blocksnet.machine_learning.strategy import ClassificationBase
from sklearn.base import BaseEstimator
import numpy as np

class SKLearnVotingClassificationStrategy(SKLearnVotingBaseStrategy, ClassificationBase):

    def __init__(self, estimators : list[tuple[str, BaseEstimator]], model_params: dict | None = None):
        super().__init__(VotingClassifier, estimators, model_params)

    def predict_proba(self, x):
        return self.model.predict_proba(x)

