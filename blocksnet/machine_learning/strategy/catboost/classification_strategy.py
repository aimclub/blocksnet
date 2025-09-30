import numpy as np
from catboost import CatBoostClassifier
from .base_strategy import CatBoostBaseStrategy
from ..classification_base import ClassificationBase


class CatBoostClassificationStrategy(ClassificationBase, CatBoostBaseStrategy):
    """CatBoostClassificationStrategy class.

    """
    def __init__(self, model_params: dict | None = None):
        """Initialize the instance.

        Parameters
        ----------
        model_params : dict | None, default: None
            Description.

        Returns
        -------
        None
            Description.

        """
        super().__init__(CatBoostClassifier, model_params)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Predict proba.

        Parameters
        ----------
        x : np.ndarray
            Description.

        Returns
        -------
        np.ndarray
            Description.

        """
        return self.model.predict_proba(x)
