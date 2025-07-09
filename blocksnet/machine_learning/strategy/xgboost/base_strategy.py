import os
import numpy as np
import pandas as pd
import xgboost as xgb
from abc import abstractmethod
from ..base_strategy import BaseStrategy

MODEL_FILENAME = "model.xgb"


class XGBoostBaseStrategy(BaseStrategy):
    def __init__(self, model_cls: type[xgb.XGBModel], model_params: dict | None = None):
        super().__init__(model_cls, model_params or {})

    def train(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        y_columns: list[str],
    ) -> float:
        super().train()
        data = self.preprocess(train_df, y_columns)
        test = self.preprocess(test_df, y_columns)
        self.model.fit(data["x"], data["y"], eval_set=[(test["x"], test["y"])], verbose=False)
        return self.model.score(test["x"], test["y"])

    def _predict(self, data: dict[str, pd.DataFrame | pd.Series]) -> np.ndarray:
        return self.model.predict(data["x"])

    def predict(self, df: pd.DataFrame, y_columns: list[str]) -> pd.DataFrame:
        super().predict()
        data = self.preprocess(df, y_columns)
        pred = self._predict(data)
        return self.postprocess(pred, df.index.tolist(), y_columns or df.columns.tolist())

    @abstractmethod
    def _get_eval_metric(self, default_metric: str) -> str:
        """Must call `super()._get_eval_metric(default_metric)` when overriden."""
        if self.model is None:
            raise ValueError("Model is not initialized")
        eval_metric = self.model.get_params().get("eval_metric", None)
        if eval_metric is None:
            if hasattr(self.model, "evals_result_") and "validation_0" in self.model.evals_result_:
                eval_metric = list(self.model.evals_result_["validation_0"].keys())[0]
            else:
                eval_metric = default_metric
        return eval_metric

    @abstractmethod
    def _validate(self, data: dict[str, pd.DataFrame | pd.Series]) -> float:
        pass

    def validate(self, df: pd.DataFrame, y_columns: list[str]) -> tuple[float, pd.DataFrame]:
        super().validate()
        data = self.preprocess(df, y_columns)
        loss = self._validate(data)
        pred = self._predict(data)
        out_df = self.postprocess(pred, df.index.tolist(), y_columns)
        return loss, out_df

    def preprocess(self, df: pd.DataFrame, y_columns: list[str] | None = None) -> dict[str, pd.DataFrame | pd.Series]:
        if y_columns is not None:
            x_columns = list(set(df.columns) - set(y_columns))
            y_columns = list(set(df.columns) & set(y_columns))
            x = df[x_columns]
            y = df[y_columns]
            y_squeezed = y.squeeze()
            if isinstance(y_squeezed, (pd.DataFrame, pd.Series)):
                return {"x": x, "y": y_squeezed}
            else:
                return {"x": x, "y": pd.Series([y_squeezed], index=y.index)}
        return {"x": df}

    def postprocess(self, pred, index: list[int], columns: list[str]) -> pd.DataFrame:
        return pd.DataFrame(pred, index=index, columns=columns)

    def _save_model(self, path: str):
        self.model.save_model(os.path.join(path, MODEL_FILENAME))

    def _load_model(self, path: str):
        model_path = os.path.join(path, MODEL_FILENAME)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        self.model = self.model_cls(**self.model_params)
        self.model.load_model(model_path)
