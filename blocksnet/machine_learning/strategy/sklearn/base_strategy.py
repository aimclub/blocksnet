from abc import ABC
import joblib
import os
from sklearn.base import BaseEstimator
from ..base_strategy import BaseStrategy

MODEL_FILENAME = "model.joblib"


class SKLearnBaseStrategy(BaseStrategy, ABC):
    """SKLearnBaseStrategy class.

    """
    def __init__(self, model_cls: type[BaseEstimator], model_params: dict | None):
        """Initialize the instance.

        Parameters
        ----------
        model_cls : type[BaseEstimator]
            Description.
        model_params : dict | None
            Description.

        Returns
        -------
        None
            Description.

        """
        super().__init__(model_cls, model_params or {})

    def _save_model(self, path: str):
        """Save model.

        Parameters
        ----------
        path : str
            Description.

        """
        joblib.dump(self.model, os.path.join(path, MODEL_FILENAME))

    def _load_model(self, path: str):
        """Load model.

        Parameters
        ----------
        path : str
            Description.

        """
        self.model = joblib.load(os.path.join(path, MODEL_FILENAME))
