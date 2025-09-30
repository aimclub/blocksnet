from abc import ABC
from ..base_strategy import SKLearnBaseStrategy
from sklearn.base import BaseEstimator
from blocksnet.machine_learning.strategy import BaseStrategy

class SKLearnEnsembleBaseStrategy(SKLearnBaseStrategy, ABC):

    """SKLearnEnsembleBaseStrategy class.

    """
    def __init__(self, model_cls : type[BaseEstimator], estimators : list[tuple[str, BaseEstimator]], model_params : dict | None):
        """Initialize the instance.

        Parameters
        ----------
        model_cls : type[BaseEstimator]
            Description.
        estimators : list[tuple[str, BaseEstimator]]
            Description.
        model_params : dict | None
            Description.

        Returns
        -------
        None
            Description.

        """
        super().__init__(model_cls, model_params)
        self._estimators = estimators

    @property
    def estimators(self) -> list[tuple[str, BaseEstimator]]:
        """Estimators.

        Returns
        -------
        list[tuple[str, BaseEstimator]]
            Description.

        """
        if self._estimators is None or len(self._estimators) == 0:
            raise ValueError("Estimators are not defined")
        return self._estimators
        
    @estimators.setter
    def estimators(self, estimators):
        """Estimators.

        Parameters
        ----------
        estimators : Any
            Description.

        """
        for name, estimator in estimators:
            if not isinstance(name, str):
                raise ValueError(f"Estimator name {name} is not a valid string")
            if not isinstance(estimator, BaseEstimator):
                raise ValueError(f"Estimator {name} is not a valid estimator")
        self._estimators = estimators

    def _load_model(self, path):
        """Load model.

        Parameters
        ----------
        path : Any
            Description.

        """
        super()._load_model(path)
        self.estimators = self.model.estimators
        
