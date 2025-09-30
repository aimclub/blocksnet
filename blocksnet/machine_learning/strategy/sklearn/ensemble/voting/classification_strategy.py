from sklearn.ensemble import VotingClassifier
from .base_strategy import SKLearnVotingBaseStrategy
from blocksnet.machine_learning.strategy import ClassificationBase
from sklearn.base import BaseEstimator
import numpy as np

class SKLearnVotingClassificationStrategy(SKLearnVotingBaseStrategy, ClassificationBase):

    """SKLearnVotingClassificationStrategy class.

    """
    def __init__(self, estimators : list[tuple[str, BaseEstimator]], model_params: dict | None = None):
        """Initialize the instance.

        Parameters
        ----------
        estimators : list[tuple[str, BaseEstimator]]
            Description.
        model_params : dict | None, default: None
            Description.

        Returns
        -------
        None
            Description.

        """
        super().__init__(VotingClassifier, estimators, model_params)

    def predict_proba(self, x):
        """Predict proba.

        Parameters
        ----------
        x : Any
            Description.

        """
        return self.model.predict_proba(x)

