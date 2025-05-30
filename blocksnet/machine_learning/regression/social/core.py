from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from .common import ModelWrapper
from .schemas import SocialIndicatorsSchema, TechnicalIndicatorsSchema


CURRENT_DIRECTORY = Path(__file__).parent
MODELS_DIRECTORY = CURRENT_DIRECTORY / "models"
MODEL_PATH = str(MODELS_DIRECTORY / "model.pickle")


class SocialRegressor(ModelWrapper):
    def __init__(self, model_path: str = MODEL_PATH, *args, **kwargs):
        ModelWrapper.__init__(self, model_path, *args, **kwargs)

    def _initialize_x(self, data: pd.DataFrame) -> pd.DataFrame:
        return TechnicalIndicatorsSchema(data)

    def _initialize_y(self, data: pd.DataFrame) -> pd.DataFrame:
        return SocialIndicatorsSchema(data)

    def get_train_data(
        self, data: pd.DataFrame, train_size: float = 0.8, drop_na: bool = True
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

        if drop_na:
            data = data.dropna(axis=1, how="all")
        x = self._initialize_x(data)
        y = self._initialize_y(data)

        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_size, random_state=42)

        return x_train, x_test, y_train, y_test

    def train(self, x_train: pd.DataFrame, y_train: pd.DataFrame, confidence_level: float = 95.0):
        self._train_model(x_train, y_train, confidence_level)

    def evaluate(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

        x = self._initialize_x(data)
        y_pred, pi_lower, pi_upper = self._evaluate_model(x)

        # Create separate DataFrames for predictions and intervals
        columns = SocialIndicatorsSchema._columns()
        pred_df = pd.DataFrame(y_pred, index=x.index, columns=columns)
        pi_lower_df = pd.DataFrame(pi_lower, index=x.index, columns=columns)
        pi_upper_df = pd.DataFrame(pi_upper, index=x.index, columns=columns)

        return pred_df, pi_lower_df, pi_upper_df

    def calculate_interval_stats(
        self, y_pred: pd.DataFrame, pi_lower_df: pd.DataFrame, pi_upper_df: pd.DataFrame, y_test: pd.DataFrame = None
    ) -> pd.DataFrame:

        stats_df = {}

        for target_name in SocialIndicatorsSchema._columns():
            # Extract lower and upper bounds from interval tuples
            pi_lower = pi_lower_df[target_name].values
            pi_upper = pi_upper_df[target_name].values
            interval_widths = pi_upper - pi_lower
            mean_width = np.mean(interval_widths)

            # Initialize metrics
            coverage = np.nan
            mse = np.nan
            rmse = np.nan
            mae = np.nan
            r2 = np.nan

            # Calculate metrics if true values are provided
            if y_test is not None and target_name in y_test.columns:
                y_values = y_test[target_name].values
                pred_values = y_pred[target_name].values

                # Coverage: percentage of true values within intervals
                coverage = np.mean((y_values >= pi_lower) & (y_values <= pi_upper)) * 100

                # Regression metrics
                mse = mean_squared_error(y_values, pred_values)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_values, pred_values)
                r2 = r2_score(y_values, pred_values)

            stats_df[target_name] = [coverage, mean_width, mse, rmse, mae, r2]

        # Create stats DataFrame
        stats_df = pd.DataFrame(
            stats_df, index=["coverage_percentage", "mean_interval_width", "mse", "rmse", "mae", "r2"]
        ).rename_axis("y")

        return stats_df.T
