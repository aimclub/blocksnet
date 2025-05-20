import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
from sklearn.utils import resample


class ModelWrapper:
    """Base class for managing a multi-output regression model."""

    def __init__(self, model_path, *args, **kwargs):
        """
        Initialize the model with a GradientBoostingRegressor wrapped in MultiOutputRegressor.

        Args:
            *args: Positional arguments for GradientBoostingRegressor.
            **kwargs: Keyword arguments for GradientBoostingRegressor.
        """
        self.base_params = kwargs
        base_model = GradientBoostingRegressor(random_state=42, **kwargs)
        self.model = MultiOutputRegressor(base_model)

        if model_path:
            self.load_model(model_path)


    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to a file.

        Args:
            filepath: Path to save the model.
        """
        joblib.dump(self.model, filepath)
        print(f"Model saved to: {filepath}")

    def load_model(self, filepath: str) -> None:
        """
        Load a trained model from a file.

        Args:
            filepath: Path to the model file.
        """
        self.model = joblib.load(filepath)
        print(f"Model loaded from: {filepath}")

    def _train_model(self, x_train: pd.DataFrame, y_train: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Train the model and compute predictions and mean squared error.

        Args:
            x_train: Training features.
            y_train: Training targets.

        Returns:
            Tuple of predictions and mean squared error.
        """
        self.model.fit(x_train, y_train)
        y_pred = self.model.predict(x_train)
        mse = mean_squared_error(y_train, y_pred, multioutput='raw_values')
        return y_pred, mse

    def _evaluate_model(self, x_test: pd.DataFrame, y_test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Evaluate the model on test data and compute predictions and mean squared error.

        Args:
            x_test: Test features.
            y_test: Test targets.

        Returns:
            Tuple of predictions and mean squared error.
        """
        y_pred = self.model.predict(x_test)
        mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
        return y_pred, mse

    def _compute_bootstrap_intervals(
        self,
        x_train: pd.DataFrame,
        y_train: pd.DataFrame,
        x_test: pd.DataFrame,
        bootstrap_loop_count: int = 200,
        alpha: float = 0.05,
        random_state: int = 42
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute confidence and prediction intervals using bootstrap.

        Args:
            x_train: Training features.
            y_train: Training targets.
            x_test: Test features.
            bootstrap_loop_count: Number of bootstrap iterations (default: 200).
            alpha: Significance level for intervals (default: 0.05).
            random_state: Random seed for reproducibility.

        Returns:
            Tuple of (mean_preds, ci_lower, ci_upper, pi_lower, pi_upper).
        """
        rng = np.random.RandomState(random_state)
        base_model = GradientBoostingRegressor(random_state=random_state, **self.base_params)
        model_full = MultiOutputRegressor(base_model).fit(x_train, y_train)
        mean_preds = model_full.predict(x_test)

        boot_preds = np.zeros((bootstrap_loop_count, x_test.shape[0], y_train.shape[1]))

        print(f"Running bootstrap ({bootstrap_loop_count} iterations)...")
        for b in tqdm(range(bootstrap_loop_count), desc="Bootstrapping"):
            x_boot, y_boot = resample(x_train, y_train, random_state=rng.randint(0, 1_000_000))
            base_model_boot = GradientBoostingRegressor(
                random_state=rng.randint(0, 1_000_000), **self.base_params
            )
            model_boot = MultiOutputRegressor(base_model_boot).fit(x_boot, y_boot)
            boot_preds[b] = model_boot.predict(x_test)

        lower_q, upper_q = 100 * (alpha / 2), 100 * (1 - alpha / 2)
        mean_boot = boot_preds.mean(axis=1)
        ci_lower = np.percentile(mean_boot, lower_q, axis=0)
        ci_upper = np.percentile(mean_boot, upper_q, axis=0)
        pi_lower = np.percentile(boot_preds, lower_q, axis=0)
        pi_upper = np.percentile(boot_preds, upper_q, axis=0)

        return mean_preds, ci_lower, ci_upper, pi_lower, pi_upper

    def _predict_with_intervals(
        self,
        x_single: pd.DataFrame,
        x_train: pd.DataFrame,
        y_train: pd.DataFrame,
        bootstrap_loop_count: int = 200,
        alpha: float = 0.05,
        random_state: int = 42
    ) -> pd.DataFrame:
        """
        Predict with bootstrap confidence and prediction intervals for a single observation.

        Args:
            x_single: Single observation to predict.
            x_train: Training features.
            y_train: Training targets.
            bootstrap_loop_count: Number of bootstrap iterations (default: 200).
            alpha: Significance level for intervals (default: 0.05).
            random_state: Random seed for reproducibility.

        Returns:
            DataFrame with predictions, confidence intervals, and prediction intervals.
        """
        if isinstance(x_single, pd.Series):
            x_single = x_single.to_frame().T

        mean, ci_lower, ci_upper, pi_lower, pi_upper = self._compute_bootstrap_intervals(
            x_train=x_train,
            y_train=y_train,
            x_test=x_single,
            bootstrap_loop_count=bootstrap_loop_count,
            alpha=alpha,
            random_state=random_state
        )

        # Ensure arrays are 1D for single-output case
        mean = np.squeeze(mean)
        pi_lower = np.squeeze(pi_lower)
        pi_upper = np.squeeze(pi_upper)

        return pd.DataFrame(
            {
                "Prediction": mean,
                "Prediction Interval": [f"[{lo:.4f}, {hi:.4f}]" for lo, hi in zip(pi_lower, pi_upper)],
                "Confidence Interval": [f"[{lo:.4f}, {hi:.4f}]" for lo, hi in zip(ci_lower, ci_upper)],
            },
            index=y_train.columns
        )