import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger
from sklearn.model_selection import train_test_split
from .common import ModelWrapper, ScalerWrapper
from .schemas import TechnicalIndicatorsSchema, SocialIndicatorsScheme
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class SocialRegressor(ModelWrapper, ScalerWrapper):

    def __init__(self, model_path=None, *args, **kwargs):
        ModelWrapper.__init__(self, model_path, *args, **kwargs)
        ScalerWrapper.__init__(self)

    def _initialize_features(self, data: pd.DataFrame, scaler, scale_data: bool) -> pd.DataFrame:
        """
        Initialize and preprocess feature data by handling infinities, NaNs, and scaling.

        Args:
            data: Input DataFrame containing features or targets.
            scaler: Scaler instance for transforming data.
            scale_data: Whether to fit the scaler to the data.

        Returns:
            Processed DataFrame with scaled features.
        """

        data = data.dropna(axis=1, how='all')

        columns = data.columns
        index = data.index

        # Scale the data if required
        if scale_data:
            logger.info("Fitting scaler to data")
            scaler.fit(data)
            data = scaler.transform(data)

        scaled_data = pd.DataFrame(data, columns=columns, index=index)
        return scaled_data.fillna(0)
    
    def get_train_data(
        self,
        data: pd.DataFrame,
        train_size: float = 0.8,
        scale_data: bool = True
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Prepare training and testing data by preprocessing and splitting.

        Args:
            data: DataFrame with technical indicators and social indicators.
            train_size: Proportion of data to use for training (default: 0.8).
            scale_data: Whether to fit the scalers to the data.

        Returns:
            Tuple of (x_train, x_test, y_train, y_test) DataFrames.
        """
        technical_indicators = TechnicalIndicatorsSchema(data)
        social_indicators = SocialIndicatorsScheme(data)
        
        x = self._initialize_features(technical_indicators, self.scaler_x, scale_data)
        y = self._initialize_features(social_indicators, self.scaler_y, scale_data)

        self.y_columns_names = y.columns.values
        # Split data into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, train_size=train_size, random_state=42
        )

        return x_train, x_test, y_train, y_test

    def train(self, x_train: pd.DataFrame, y_train: pd.DataFrame, confidence_level: float = 95.0):
        """
        Train the model with quantile regression for median, lower, and upper quantiles.

        Args:
            x_train: Training features.
            y_train: Training targets.
            confidence_level: Desired confidence level for prediction intervals as a percentage (default: 95.0 for 95%).

        """
        self._train_model(x_train, y_train, confidence_level)

    def evaluate(self, x_test: pd.DataFrame) -> tuple[np.ndarray]:
        """
        Evaluate the model on test data using median predictions.

        Args:
            x_test: Test features.

        Returns:
            Tuple of median predictions and mean squared error.
        """
        return self._evaluate_model(x_test)

    def predict(self, x: pd.DataFrame, inverse_transform: bool = False) -> pd.DataFrame:
        """
        Predict median values (quantile 0.5) for the input data.

        Args:
            x: Input features.
            inverse_transform: Whether to apply inverse scaling to predictions.

        Returns:
            DataFrame with median predictions.
        """
        if isinstance(x, pd.Series):
            x = x.to_frame().T
        x = self._initialize_features(x, self.scaler_x, scale_data=False)
        predictions = self._predict_median(x)
        if inverse_transform:
            predictions = self.inverse_transform_y(predictions)
        return pd.DataFrame(predictions, columns=self.model_median.estimators_[0].feature_names_out_, index=x.index)


    def predict_with_intervals(
        self, x: pd.DataFrame, inverse_transform: bool = False
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Predict with median and prediction intervals using pre-trained quantile models.

        Args:
            x: Input features.
            inverse_transform: Whether to apply inverse scaling to predictions.

        Returns:
            Tuple of two DataFrames:
            1. DataFrame with columns '{target_name}' for median predictions
            2. DataFrame with columns '{target_name}' for prediction intervals in format '[lower, upper]'
        """
        if isinstance(x, pd.Series):
            x = x.to_frame().T
        x = self._initialize_features(x, self.scaler_x, scale_data=False)
        prediction, pi_lower, pi_upper = self._predict_with_intervals(x)
        
        if inverse_transform:
            prediction = self.inverse_transform_y(prediction)
            pi_lower = self.inverse_transform_y(pi_lower)
            pi_upper = self.inverse_transform_y(pi_upper)

        # Create separate DataFrames for predictions and intervals
        pred_df = {}
        pi_df = {}
        
        for idx, target_name in enumerate(self.y_columns_names):
            pred_df[target_name] = prediction[:, idx]
            pi_df[target_name] = [
                f"[{lo:.4f}, {hi:.4f}]" for lo, hi in zip(pi_lower[:, idx], pi_upper[:, idx])
            ]
        
        # Create DataFrames
        pred_df = pd.DataFrame(pred_df, index=x.index)
        pi_df = pd.DataFrame(pi_df, index=x.index)

        return pred_df.round(3), pi_df

    def calculate_interval_stats(
        self, pred_df: pd.DataFrame, pi_df: pd.DataFrame, y_true: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        Calculate statistics and metrics for prediction intervals.

        Args:
            pred_df: DataFrame with columns '{target_name}' for median predictions.
            pi_df: DataFrame with columns '{target_name}' for prediction intervals in format '[lower, upper]'.
            y_true: Optional true values to calculate coverage percentage and regression metrics.

        Returns:
            DataFrame with index 'y' containing coverage percentage, mean interval width,
            MSE, RMSE, MAE, and R² for each target.
        """
        stats_df = {}
        
        for target_name in self.y_columns_names:
            # Extract lower and upper bounds from interval strings
            intervals = pi_df[target_name].str.extract(r'\[([-\d.]+),\s*([-\d.]+)\]').astype(float)
            pi_lower = intervals[0].values
            pi_upper = intervals[1].values
            interval_widths = pi_upper - pi_lower
            mean_width = np.mean(interval_widths)
            
            # Initialize metrics
            coverage = np.nan
            mse = np.nan
            rmse = np.nan
            mae = np.nan
            r2 = np.nan
            
            # Calculate metrics if true values are provided
            if y_true is not None and target_name in y_true.columns:
                y_values = y_true[target_name].values
                pred_values = pred_df[target_name].values
                
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
            stats_df,
            index=['coverage_percentage', 'mean_interval_width', 'mse', 'rmse', 'mae', 'r2']
        ).rename_axis('y')
        
        return stats_df.T.round(3)