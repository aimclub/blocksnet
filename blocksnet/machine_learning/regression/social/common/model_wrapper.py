import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error

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
       
        self.base_params = kwargs
        print(self.base_params)
        self.model_median = MultiOutputRegressor(
            GradientBoostingRegressor(random_state=42, **kwargs)
        )
        self.model_lower = None  # To be initialized during training
        self.model_upper = None  # To be initialized during training
        if model_path:
            self.load_model(model_path)

    def save_model(self, filepath: str) -> None:
        """
        Save the trained models (median, lower, upper) to a file.

        Args:
            filepath: Path to save the models.
        """
        joblib.dump({
            'model_median': self.model_median,
            'model_lower': self.model_lower,
            'model_upper': self.model_upper
        }, filepath)
        print(f"Models saved to: {filepath}")

    def load_model(self, filepath: str) -> None:
        """
        Load trained models from a file.

        Args:
            filepath: Path to the model file.
        """
        models = joblib.load(filepath)
        self.model_median = models['model_median']
        self.model_lower = models['model_lower']
        self.model_upper = models['model_upper']
        print(f"Models loaded from: {filepath}")

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

        # Train median model (quantile 0.5)
        self.model_median.fit(x_train, y_train)

        # Train lower quantile model
        self.model_lower = MultiOutputRegressor(
            GradientBoostingRegressor(
                loss='quantile',
                alpha=lower_quantile,
                random_state=42,
                **self.base_params
            )
        )
        self.model_lower.fit(x_train, y_train)

        # Train upper quantile model
        self.model_upper = MultiOutputRegressor(
            GradientBoostingRegressor(
                loss='quantile',
                alpha=upper_quantile,
                random_state=42,
                **self.base_params
            )
        )
        self.model_upper.fit(x_train, y_train)

    def _evaluate_model(self, x_test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Evaluate the model on test data using median predictions.

        Args:
            x_test: Test features.
            y_test: Test targets.

        Returns:
            Tuple of median predictions and mean squared error.
        """
        y_pred = self.model_median.predict(x_test)

        return y_pred

    def _predict_median(self, x: pd.DataFrame) -> np.ndarray:
        """
        Predict median values using the pre-trained median model.

        Args:
            x: Input features.

        Returns:
            Array of median predictions.
        """
        return self.model_median.predict(x)

    def _predict_with_intervals(self, x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict median and prediction intervals using pre-trained quantile models.

        Args:
            x: Input features.

        Returns:
            Tuple of (median predictions, lower PI, upper PI).
        """
        mean_preds = self.model_median.predict(x)
        pi_lower = self.model_lower.predict(x)
        pi_upper = self.model_upper.predict(x)
        return mean_preds, pi_lower, pi_upper