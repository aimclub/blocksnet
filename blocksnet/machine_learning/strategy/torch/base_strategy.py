import os
from abc import abstractmethod
import joblib
import torch
from ..base_strategy import BaseStrategy

MODEL_FILENAME = "model.pt"
SCALER_FILENAME = "scaler.joblib"


class TorchBaseStrategy(BaseStrategy):
    def __init__(self, model_cls: type[torch.nn.Module], model_params: dict | None = None, scaler=None, device=None):
        super().__init__(model_cls, model_params or {})
        self.scaler = scaler
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def _build_optimizer(self, optimizer_cls: type[torch.optim.Optimizer], optimizer_params: dict):
        """Must call super()._build_optimizer() when overriden."""
        return optimizer_cls(**optimizer_params)

    @abstractmethod
    def _build_criterion(self, criterion_cls: type[torch.nn.Module], criterion_params: dict):
        """Must call super()._build_criterion() when overriden."""
        return criterion_cls(**criterion_params)

    def _save_model(self, path: str):
        state_dict = self.model.state_dict()
        torch.save(state_dict, os.path.join(path, MODEL_FILENAME))

    def _load_model(self, path: str):
        model_path = os.path.join(path, MODEL_FILENAME)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        state_dict = torch.load(model_path)
        self._model = self.model_cls(**self.model_params)
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
