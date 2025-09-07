from sklearn.ensemble import VotingRegressor
from .base_strategy import SKLearnVotingBaseStrategy
from sklearn.base import BaseEstimator


class SKLearnVotingRegressionStrategy(SKLearnVotingBaseStrategy):

    def __init__(self, estimators : list[tuple[str, BaseEstimator]], model_params: dict | None = None):
        super().__init__(VotingRegressor, estimators, model_params)
