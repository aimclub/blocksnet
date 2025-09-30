import xgboost as xgb
import numpy as np
from .base_strategy import XGBoostBaseStrategy


class XGBoostRegressionStrategy(XGBoostBaseStrategy):
    """XGBoostRegressionStrategy class.

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
        super().__init__(xgb.XGBRegressor, model_params)
