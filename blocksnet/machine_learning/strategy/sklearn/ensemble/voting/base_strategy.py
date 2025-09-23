from abc import ABC
from ..base_strategy import SKLearnEnsembleBaseStrategy
import numpy as np

class SKLearnVotingBaseStrategy(SKLearnEnsembleBaseStrategy, ABC):

    def train(self, x_train : np.ndarray, y_train : np.ndarray, x_test : np.ndarray, y_test : np.ndarray) -> float:
        self.model = self.model_cls(estimators=self.estimators, **self.model_params)
        self.model.fit(x_train, y_train)
        return self.model.score(x_test, y_test)
    
    def predict(self, x):
        return self.model.predict(x)