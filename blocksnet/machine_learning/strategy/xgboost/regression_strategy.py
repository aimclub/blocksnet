import xgboost as xgb
import pandas as pd
from .base_strategy import XGBoostBaseStrategy


class XGBoostRegressionStrategy(XGBoostBaseStrategy):
    def __init__(self, model_params: dict | None = None):
        super().__init__(xgb.XGBRegressor, model_params)

    def _get_eval_metric(self) -> str:
        return super()._get_eval_metric("rmse")

    def _validate(self, data: dict[str, pd.DataFrame | pd.Series]) -> float:
        y_true = data["y"].to_numpy()
        y_pred = self._predict(data)
        eval_metric = self._get_eval_metric()

        if eval_metric == "rmse":
            return ((y_pred - y_true) ** 2).mean() ** 0.5
        elif eval_metric == "mse":
            return ((y_pred - y_true) ** 2).mean()
        elif eval_metric == "mae":
            return abs(y_pred - y_true).mean()
        else:
            raise NotImplementedError(f"Unsupported eval_metric '{eval_metric}' for regression.")
