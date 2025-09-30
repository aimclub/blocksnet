from abc import ABC
from ..base_strategy import SKLearnEnsembleBaseStrategy
import numpy as np

class SKLearnVotingBaseStrategy(SKLearnEnsembleBaseStrategy, ABC):

    """SKLearnVotingBaseStrategy class.

    """
    def train(self, x_train : np.ndarray, y_train : np.ndarray, x_test : np.ndarray, y_test : np.ndarray) -> float:
        """Train.

        Parameters
        ----------
        x_train : np.ndarray
            Description.
        y_train : np.ndarray
            Description.
        x_test : np.ndarray
            Description.
        y_test : np.ndarray
            Description.

        Returns
        -------
        float
            Description.

        """
        self.model = self.model_cls(estimators=self.estimators, **self.model_params)
        self.model.fit(x_train, y_train)
        return self.model.score(x_test, y_test)
    
    def predict(self, x):
        """Predict.

        Parameters
        ----------
        x : Any
            Description.

        """
        return self.model.predict(x)
