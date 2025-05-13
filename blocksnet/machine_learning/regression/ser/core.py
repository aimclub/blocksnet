
from sklearn.model_selection import train_test_split
import pandas as pd

import numpy as np

from loguru import logger
from .common import ModelWrapper, ScalerWrapper
from .schemas import TechnicalIndicatorsSchema, SotialIndicatorsSchema
import numpy as np
import matplotlib.pyplot as plt


class IndicatorRegressor(ModelWrapper, ScalerWrapper):
    def __init__(self, *args, **kwargs):
        ModelWrapper.__init__(self, *args, **kwargs)
        ScalerWrapper.__init__(self)

    def _initialize_x(self, tech_indincators: pd.DataFrame, fit_scaler: bool):
        X = pd.DataFrame(tech_indincators) if not isinstance(tech_indincators, pd.DataFrame) else tech_indincators
        float_max = np.finfo(np.float64).max
        float_min = np.finfo(np.float64).min

        for col in X.columns:
            if np.issubdtype(X[col].dtype, np.number):
                X[col] = X[col].replace(np.inf, float_max)
                X[col] = X[col].replace(-np.inf, float_min)
                X[col] = X[col].fillna(0)

        X = X.dropna(axis=1, how='all')
        zero_percentage = ((X == 0.0) | X.isna()).mean()
        X = X.loc[:, zero_percentage <= 0.2]
        X = X.replace([np.inf, -np.inf], np.nan).dropna()
        # Масштабирование Y (indicators)

        X_columns = X.columns
        X_index = X.index

        if fit_scaler:
            logger.info("Fitting the scaler")
            self.scaler_X.fit(X)
        X = self.scaler_X.fit_transform(X)

        X = pd.DataFrame(X, columns=X_columns, index=X_index)

        return X

    def _initialize_y(self, social_indicators: pd.DataFrame, fit_scaler: bool):
        Y = pd.DataFrame(social_indicators) if not isinstance(social_indicators, pd.DataFrame) else social_indicators
        float_max = np.finfo(np.float64).max
        float_min = np.finfo(np.float64).min

        for col in Y.columns:
            if np.issubdtype(Y[col].dtype, np.number):
                Y[col] = Y[col].replace(np.inf, float_max)
                Y[col] = Y[col].replace(-np.inf, float_min)
                Y[col] = Y[col].fillna(0)

        Y = Y.dropna(axis=1, how='all')
        zero_percentage = ((Y == 0.0) | Y.isna()).mean()
        Y = Y.loc[:, zero_percentage <= 0.2]
        Y = Y.replace([np.inf, -np.inf], np.nan).dropna()

        Y_columns = Y.columns
        Y_index = Y.index

        if fit_scaler:
            logger.info("Fitting the scaler")
            self.scaler_Y.fit(Y)
        Y = self.scaler_Y.fit_transform(Y) 
        Y = pd.DataFrame(Y, columns=Y_columns, index=Y_index)


        return Y

    def get_train_data(
        self, tech_indincators: pd.DataFrame,
          social_indicators: pd.DataFrame, train_size: float = 0.8, fit_scaler: bool = True):
        x = self._initialize_x(tech_indincators, fit_scaler)
        y = self._initialize_y(social_indicators, fit_scaler)

        combined = pd.concat([x, y], axis=1)
        filtered = combined[~combined.gt(8.988466e+304).any(axis=1)]
        x = filtered[x.columns]
        y = filtered[y.columns]

        # Разделение данных
        X_train, X_test, y_train, y_test = train_test_split(
                x, y, train_size=train_size, random_state=42
            )
        y_train = y_train.fillna(0)
        X_train = X_train.fillna(0)

        y_test = y_test.fillna(0)
        X_test = X_test.fillna(0)

        return X_train, X_test, y_train, y_test

    def train(self, X_train, y_train):
        return self._train_model(X_train, y_train)

    def validate(self, X_test, y_test):
        return self._evaluate_model(X_test, y_test)
    
    def compute_bootstrap_intervals(self, X_train, y_train, X_test, B=200, alpha=0.05, random_state=42):
        return self._compute_bootstrap_intervals(X_train, y_train, X_test, B=B, alpha=alpha, random_state=random_state)

    def plot_prediction_intervals(self, mean_preds, pi_lower, pi_upper, y_test, column_names, inverse_transform=False):
        """
        Строит графики предсказаний с предсказательными интервалами (PI).
        
        Параметры:
        - mean_preds: np.ndarray, shape = (n_samples, n_outputs)
        - pi_lower: np.ndarray, нижняя граница PI
        - pi_upper: np.ndarray, верхняя граница PI
        - y_test: np.ndarray, истинные значения
        - column_names: список названий выходов
        - scaler: экземпляр ScalerWrapper (если inverse_transform=True)
        - inverse_transform: логический флаг, применять ли обратное преобразование к Y
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

            plt.plot(np.arange(n_samples), mean_preds[:n_samples, i], label='Точечные предсказания', color='blue')
            plt.plot(np.arange(n_samples), y_test[:n_samples, i], label='Истинные значения', color='red', linestyle='--')
            plt.fill_between(np.arange(n_samples), pi_lower[:n_samples, i], pi_upper[:n_samples, i], 
                            color='lightblue', alpha=0.5, label='95% PI')

            plt.title(f'Предсказания и 95% PI для {col_name}')
            plt.xlabel('Номер образца')
            plt.ylabel(col_name)
            plt.legend()

        plt.tight_layout()
        plt.show()


    
