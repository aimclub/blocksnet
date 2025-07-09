import os
from abc import abstractmethod
import joblib
import torch
import pandas as pd
from tqdm import tqdm
from ..base_strategy import BaseStrategy

TRAIN_LOSS_TEXT = "Train loss"
TEST_LOSS_TEXT = "Test loss"

MODEL_FILENAME = "model.pt"
SCALER_FILENAME = "scaler.joblib"

OPTIMIZER_CLS = torch.optim.Adam
OPTIMIZER_PARAMS = {"lr": 3e-4, "weight_decay": 1e-5}
CRITERION_CLS = torch.nn.MSELoss
CRITERION_PARAMS = {}


class TorchBaseStrategy(BaseStrategy):
    def __init__(self, model_cls: type[torch.nn.Module], model_params: dict | None = None, scaler=None, device=None):
        super().__init__(model_cls, model_params or {})
        self.scaler = scaler
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_optimizer(
        self, optimizer_cls: type[torch.optim.Optimizer] | None = None, optimizer_params: dict | None = None
    ):
        if optimizer_cls is None:
            optimizer_cls = OPTIMIZER_CLS
        if optimizer_params is None:
            optimizer_params = OPTIMIZER_PARAMS
        return optimizer_cls(self.model.parameters(), **optimizer_params)

    def _build_criterion(
        self, criterion_cls: type[torch.nn.Module] | None = None, criterion_params: dict | None = None
    ):
        if criterion_cls is None:
            criterion_cls = CRITERION_CLS
        if criterion_params is None:
            criterion_params = CRITERION_PARAMS
        return criterion_cls(**criterion_params)

    def _save_model(self, path: str):
        state_dict = self.model.state_dict()
        torch.save(state_dict, os.path.join(path, MODEL_FILENAME))

    def _load_model(self, path: str):
        model_path = os.path.join(path, MODEL_FILENAME)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        state_dict = torch.load(model_path)
        self.model = self.model_cls(**self.model_params)
        self.model.load_state_dict(state_dict)

    def _save_scaler(self, path: str):
        if self.scaler is not None:
            joblib.dump(self.scaler, os.path.join(path, SCALER_FILENAME))

    def _load_scaler(self, path: str):
        self.scaler = None
        scaler_path = os.path.join(path, SCALER_FILENAME)
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)

    def save(self, path: str):
        super().save(path)
        self._save_scaler(path)

    def load(self, path: str):
        super().load(path)
        self._load_scaler(path)
