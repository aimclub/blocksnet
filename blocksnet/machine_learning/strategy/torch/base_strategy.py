import os
from abc import abstractmethod
import joblib
import torch
import numpy as np
from ..base_strategy import BaseStrategy

MODEL_FILENAME = "model.pt"
SCALER_FILENAME_POSTFIX = "scaler.joblib"


class TorchBaseStrategy(BaseStrategy):
    def __init__(
        self,
        model_cls: type[torch.nn.Module],
        model_params: dict | None = None,
        scalers: dict | None = None,
        device=None,
    ):
        super().__init__(model_cls, model_params or {})
        self.scalers = scalers or {}
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def _build_optimizer(self, optimizer_cls: type[torch.optim.Optimizer], optimizer_params: dict):
        """Must call super()._build_optimizer() when overriden."""
        return optimizer_cls(self.model.parameters(), **optimizer_params)

    @abstractmethod
    def _build_criterion(self, criterion_cls: type[torch.nn.Module], criterion_params: dict):
        """Must call super()._build_criterion() when overriden."""
        return criterion_cls(**criterion_params)

    def _build_model(self, input_size: int, output_size: int, **kwargs) -> torch.nn.Module:
        model_params = {**self.model_params, "input_size": input_size, "output_size": output_size, **kwargs}
        model = self.model_cls(**model_params)
        self._model = model
        self.model_params = model_params
        return model

    def _preprocess(self, key: str, fit: bool, *arrays: np.ndarray) -> tuple[torch.Tensor, ...]:
        scaler = self.scalers.get(key)
        arrs = list(arrays)
        if scaler is not None:
            if fit:
                scaler.fit(arrs[0])
            arrs = [scaler.transform(a) for a in arrs]
        arrs = [torch.tensor(a, dtype=torch.float32, device=self.device) for a in arrs]
        return tuple(arrs)

    def _postprocess(self, key: str, *arrays: torch.Tensor) -> tuple[np.ndarray, ...]:
        scaler = self.scalers.get(key)
        arrs = list(arrays)
        arrs = [a.cpu().numpy() for a in arrs]
        if scaler is not None:
            arrs = [scaler.inverse_transform(a) for a in arrs]
        return tuple(arrs)

    def _save_model(self, path: str):
        state_dict = self.model.to("cpu").state_dict()
        torch.save(state_dict, os.path.join(path, MODEL_FILENAME))

    def _load_model(self, path: str):
        model_path = os.path.join(path, MODEL_FILENAME)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        state_dict = torch.load(model_path)
        self._model = self.model_cls(**self.model_params)
        self.model.load_state_dict(state_dict)

    def _save_scalers(self, path: str):
        for name, scaler in self.scalers.items():
            if scaler is not None:
                filename = os.path.join(path, f"{name}_{SCALER_FILENAME_POSTFIX}")
                joblib.dump(scaler, filename)

    def _load_scalers(self, path: str):
        self.scalers = {}
        for fname in os.listdir(path):
            if fname.endswith(f"_{SCALER_FILENAME_POSTFIX}"):
                key = fname[: -len(f"_{SCALER_FILENAME_POSTFIX}")]
                full_path = os.path.join(path, fname)
                self.scalers[key] = joblib.load(full_path)

    def save(self, path: str):
        super().save(path)
        self._save_scalers(path)

    def load(self, path: str):
        super().load(path)
        self._load_scalers(path)
