import torch
import pandas as pd
from tqdm import tqdm
from .base_strategy import TorchBaseStrategy


class TorchRegresionStrategy(TorchBaseStrategy):
    def _build_optimizer(
        self, optimizer_cls: type[torch.optim.Optimizer] | None = None, optimizer_params: dict | None = None
    ):
        optimizer_cls = optimizer_cls or torch.optim.Adam
        optimizer_params = optimizer_params or {"lr": 3e-4, "weight_decay": 1e-5}
        return super()._build_optimizer(optimizer_cls, optimizer_params)

    def _build_criterion(
        self, criterion_cls: type[torch.nn.Module] | None = None, criterion_params: dict | None = None
    ):
        criterion_cls = criterion_cls or torch.nn.MSELoss
        criterion_params = criterion_params or {}
        return super()._build_criterion(criterion_cls, criterion_params)

    def predict(self, df: pd.DataFrame, y_columns: list[str] | None = None) -> pd.DataFrame:
        super().predict()
