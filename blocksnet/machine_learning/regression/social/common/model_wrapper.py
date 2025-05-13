from .....config import log_config
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import joblib
from tqdm import tqdm
from sklearn.utils import resample

class ModelWrapper:
    def __init__(self,  *args, **kwargs):
        self.base_params = kwargs
        gbr = GradientBoostingRegressor(random_state=42, **kwargs)
        self.model  = MultiOutputRegressor(gbr)

    def save_model(self, filepath: str):
        """Сохраняет модель в указанный файл."""
        joblib.dump(self.model, filepath)
        print(f"Модель сохранена в: {filepath}")
    
    def load_model(self, filepath: str):
        """Загружает модель из файла."""
        self.model = joblib.load(filepath)
        print(f"Модель загружена из: {filepath}")

    def _train_model(self, X_train: pd.DataFrame, y_train: pd.DataFrame):
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_train)
        mse = mean_squared_error(y_train, y_pred, multioutput='raw_values')
        print(f'Средняя MSE: {np.mean(mse)}')
        return y_pred, mse

    def _evaluate_model(self, X_test: pd.DataFrame, y_test: pd.DataFrame):
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
        print(f'Средняя MSE: {np.mean(mse)}')
        return y_pred, mse

    def _compute_bootstrap_intervals(self, X_train, y_train, X_test, B=200, alpha=0.05, random_state=42):
        """
        Вычисляет доверительные и предиктивные интервалы бутстрэпом.
        Возвращает:
          - mean_preds: точечные предсказания на модели, обученной на всех данных
          - ci_lower, ci_upper: доверительные интервалы среднего предсказания
          - pi_lower, pi_upper: предиктивные интервалы
        """
        rng = np.random.RandomState(random_state)
        base = GradientBoostingRegressor(random_state=random_state, **self.base_params)
        model_full = MultiOutputRegressor(base).fit(X_train, y_train)
        mean_preds = model_full.predict(X_test)

        boot_preds = np.zeros((B, X_test.shape[0], y_train.shape[1]))

        print(f"Выполняется бутстрэп ({B} повторов)...")
        for b in tqdm(range(B), desc="Bootstrapping"):
            Xb, yb = resample(X_train, y_train, random_state=rng.randint(0, 1e6))
            base_b = GradientBoostingRegressor(random_state=rng.randint(0, 1e6), **self.base_params)
            model_b = MultiOutputRegressor(base_b).fit(Xb, yb)
            boot_preds[b] = model_b.predict(X_test)

        lower_q, upper_q = 100 * (alpha / 2), 100 * (1 - alpha / 2)
        mean_boot = boot_preds.mean(axis=1)

        ci_lower = np.percentile(mean_boot, lower_q, axis=0)
        ci_upper = np.percentile(mean_boot, upper_q, axis=0)
        pi_lower = np.percentile(boot_preds, lower_q, axis=0)
        pi_upper = np.percentile(boot_preds, upper_q, axis=0)

        return mean_preds, ci_lower, ci_upper, pi_lower, pi_upper

