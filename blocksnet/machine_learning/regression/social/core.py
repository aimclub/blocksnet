import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger
from sklearn.model_selection import train_test_split
from .common import ModelWrapper, ScalerWrapper
from .schemas import TechnicalIndicatorsSchema

class SocialRegressor(ModelWrapper, ScalerWrapper):

    def __init__(self,  model_path=None, *args, **kwargs):
        ModelWrapper.__init__(self, model_path, *args, **kwargs)
        ScalerWrapper.__init__(self)

    def _initialize_features(self, data: pd.DataFrame, scaler, fit_scaler: bool) -> pd.DataFrame:
        """
        Initialize and preprocess feature data by handling infinities, NaNs, and scaling.

        Args:
            data: Input DataFrame containing features or targets.
            scaler: Scaler instance for transforming data.
            fit_scaler: Whether to fit the scaler to the data.

        Returns:
            Processed DataFrame with scaled features.
        """
        data = pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data
        float_max = np.finfo(np.float64).max
        float_min = np.finfo(np.float64).min

        # Handle infinities and NaNs in numeric columns
        for col in data.columns:
            if np.issubdtype(data[col].dtype, np.number):
                data[col] = data[col].replace([np.inf, -np.inf], [float_max, float_min]).fillna(0)

        # Drop columns with all NaNs and those with excessive zeros
        data = data.dropna(axis=1, how='all')
        zero_percentage = (data == 0.0) | data.isna()
        data = data.loc[:, zero_percentage.mean() <= 0.2]
        data = data.replace([np.inf, -np.inf], np.nan).dropna()

        columns = data.columns
        index = data.index

        # Scale the data if required
        if fit_scaler:
            logger.info("Fitting scaler to data")
            scaler.fit(data)
        scaled_data = scaler.transform(data)

        df = pd.DataFrame(scaled_data, columns=columns, index=index)
        return df.fillna(0)
    
    def get_train_data(
        self,
        technical_indicators: pd.DataFrame,
        social_indicators: pd.DataFrame,
        train_size: float = 0.8,
        fit_scaler: bool = True
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Prepare training and testing data by preprocessing and splitting.

        Args:
            technical_indicators: DataFrame with technical indicators (features).
            social_indicators: DataFrame with social indicators (targets).
            train_size: Proportion of data to use for training (default: 0.8).
            fit_scaler: Whether to fit the scalers to the data.

        Returns:
            Tuple of (x_train, x_test, y_train, y_test) DataFrames.
        """
        technical_indicators = TechnicalIndicatorsSchema(technical_indicators)
        x = self._initialize_features(technical_indicators, self.scaler_X, fit_scaler)
        y = self._initialize_features(social_indicators, self.scaler_Y, fit_scaler)

        # Split data into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, train_size=train_size, random_state=42
        )

        return x_train, x_test, y_train, y_test

    def train(self, x_train: pd.DataFrame, y_train: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Train the model on the provided data.

        Args:
            x_train: Training features.
            y_train: Training targets.

        Returns:
            Tuple of predictions and mean squared error.
        """
        return self._train_model(x_train, y_train)

    def evaluate(self, x_test: pd.DataFrame, y_test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Evaluate the model on the test data.

        Args:
            x_test: Test features.
            y_test: Test targets.

        Returns:
            Tuple of predictions and mean squared error.
        """
        return self._evaluate_model(x_test, y_test)

    def compute_bootstrap_intervals(
        self,
        x_train: pd.DataFrame,
        y_train: pd.DataFrame,
        x_test: pd.DataFrame,
        bootstrap_loop_count: int = 200,
        alpha: float = 0.05,
        random_state: int = 42
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute bootstrap confidence and prediction intervals.

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
        return self._compute_bootstrap_intervals(
            x_train, y_train, x_test, bootstrap_loop_count, alpha, random_state
        )

    def predict_with_intervals(
        self,
        x_single: pd.DataFrame,
        x_train: pd.DataFrame,
        y_train: pd.DataFrame,
        bootstrap_loop_count: int = 200,
        alpha: float = 0.05,
        random_state: int = 42
    ) -> pd.DataFrame:
        """
        Predict with confidence and prediction intervals for a single observation.

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
        return self._predict_with_intervals(
            x_single, x_train, y_train, bootstrap_loop_count, alpha, random_state
        )

    def plot_prediction_intervals(
        self,
        mean_preds: np.ndarray,
        pi_lower: np.ndarray,
        pi_upper: np.ndarray,
        y_test: np.ndarray,
        column_names: list[str],
        inverse_transform: bool = False
    ) -> None:
        """
        Plot predictions with prediction intervals (PI).

        Args:
            mean_preds: Predicted values (shape: [n_samples, n_outputs]).
            pi_lower: Lower bounds of prediction intervals.
            pi_upper: Upper bounds of prediction intervals.
            y_test: True target values.
            column_names: List of target column names.
            inverse_transform: Whether to apply inverse scaling to predictions and targets.
        """
        if inverse_transform:
            mean_preds = self.inverse_transform_Y(mean_preds)
            pi_lower = self.inverse_transform_Y(pi_lower)
            pi_upper = self.inverse_transform_Y(pi_upper)
            y_test = self.inverse_transform_Y(y_test)

        n_samples = min(mean_preds.shape[0], 100)
        plt.figure(figsize=(12, 6 * len(column_names)))

        for i, col_name in enumerate(column_names):
            plt.subplot(len(column_names), 1, i + 1)
            plt.plot(np.arange(n_samples), mean_preds[:n_samples, i], label='Predicted Values', color='blue')
            plt.plot(np.arange(n_samples), y_test[:n_samples, i], label='True Values', color='red', linestyle='--')
            plt.fill_between(
                np.arange(n_samples),
                pi_lower[:n_samples, i],
                pi_upper[:n_samples, i],
                color='lightblue',
                alpha=0.5,
                label='95% Prediction Interval'
            )
            plt.title(f'Predictions and 95% PI for {col_name}')
            plt.xlabel('Sample Index')
            plt.ylabel(col_name)
            plt.legend()

        plt.tight_layout()
        plt.show()