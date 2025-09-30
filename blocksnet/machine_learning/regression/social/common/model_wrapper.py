import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from loguru import logger

MODEL_MEDIAN_KEY = "model_median"
MODEL_LOWER_KEY = "model_lower"
MODEL_UPPER_KEY = "model_upper"
MODEL_PARAMS_KEY = "model_params"


class ModelWrapper:
    """Base class for managing a multi-output regression model with quantile regression."""

    def __init__(self, model_path, *args, **kwargs):
        """
        Initialize the model with GradientBoostingRegressor for median and quantiles.

        Args:
            model_path: Path to a saved model file (optional).
            *args: Positional arguments for GradientBoostingRegressor.
            **kwargs: Keyword arguments for GradientBoostingRegressor.
        """

        self.model_lower = None
        self.model_median = None
        self.model_upper = None
        self.model_params = None

        if len(kwargs) > 0:
            logger.warning("Keyword arguments provided. The model_path is ignored")
            self.model_params = kwargs
        else:
            self.load_model(model_path)

    def save_model(self, filepath: str) -> None:
        """
        Save the trained models (median, lower, upper) to a file.

        Args:
            filepath: Path to save the models.
        """
        joblib.dump(
            {
                MODEL_MEDIAN_KEY: self.model_median,
                MODEL_LOWER_KEY: self.model_lower,
                MODEL_UPPER_KEY: self.model_upper,
                MODEL_PARAMS_KEY: self.model_params,
            },
            filepath,
        )

    def load_model(self, filepath: str) -> None:
        """
        Load trained models from a file.

        Args:
            filepath: Path to the model file.
        """
        data = joblib.load(filepath)
        self.model_median = data[MODEL_MEDIAN_KEY]
        self.model_lower = data[MODEL_LOWER_KEY]
        self.model_upper = data[MODEL_UPPER_KEY]
        self.model_params = data[MODEL_PARAMS_KEY]

    def _train_model(self, x_train: pd.DataFrame, y_train: pd.DataFrame, confidence_level: float = 95.0):
        """
        Train models for median, lower, and upper quantiles.

        Args:
            x_train: Training features.
            y_train: Training targets.
            confidence_level: Desired confidence level for prediction intervals as a percentage (default: 95.0 for 95%).

        Returns:
            Tuple of median predictions and mean squared error.
        """
        alpha = 1 - (confidence_level / 100)
        lower_quantile = alpha / 2
        upper_quantile = 1 - alpha / 2

        logger.info("Fitting median model")
        self.model_median = MultiOutputRegressor(GradientBoostingRegressor(random_state=42, **self.model_params))
        self.model_median.fit(x_train, y_train)

        logger.info("Fitting lower model")
        self.model_lower = MultiOutputRegressor(
            GradientBoostingRegressor(loss="quantile", alpha=lower_quantile, random_state=42, **self.model_params)
        )
        self.model_lower.fit(x_train, y_train)

        logger.info("Fitting upper model")
        self.model_upper = MultiOutputRegressor(
            GradientBoostingRegressor(loss="quantile", alpha=upper_quantile, random_state=42, **self.model_params)
        )
        self.model_upper.fit(x_train, y_train)

    def _evaluate_model(self, x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate model.

        Parameters
        ----------
        x : pd.DataFrame
            Description.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            Description.

        """
        if self.model_median is None:
            raise KeyError("The model is not fitted")
        preds = self.model_median.predict(x)
        pi_lower = self.model_lower.predict(x)
        pi_upper = self.model_upper.predict(x)
        return preds, pi_lower, pi_upper
