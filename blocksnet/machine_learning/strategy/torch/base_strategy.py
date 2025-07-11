import os
from abc import ABC, abstractmethod
from typing import Any
import joblib
import torch
import numpy as np
from tqdm import tqdm
from blocksnet.config import log_config
from ..base_strategy import BaseStrategy

MODEL_FILENAME = "model.pt"
SCALER_FILENAME_POSTFIX = "scaler.joblib"

TRAIN_LOSS_TEXT = "Train loss"
TEST_LOSS_TEXT = "Test loss"


class _TorchBaseStrategy(BaseStrategy, ABC):
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

    def _build_optimizer(
        self, optimizer_cls: type[torch.optim.Optimizer] | None, optimizer_params: dict | None
    ) -> torch.optim.Optimizer:
        optimizer_cls = optimizer_cls or torch.optim.Adam
        optimizer_params = optimizer_params or {"lr": 1e-3, "weight_decay": 1e-4}
        return optimizer_cls(self.model.parameters(), **optimizer_params)

    def _scale(self, key: str, fit: bool, *arrays: np.ndarray) -> tuple[np.ndarray, ...]:
        scaler = self.scalers.get(key)
        arrs = list(arrays)
        if scaler is not None:
            if fit:
                scaler.fit(arrs[0])
            arrs = [scaler.transform(a) for a in arrs]
        # arrs = [torch.tensor(a, dtype=torch.float32, device=self.device) for a in arrs]
        return tuple(arrs)

    def _inverse_scale(self, key: str, *arrays: np.ndarray) -> tuple[np.ndarray, ...]:
        scaler = self.scalers.get(key)
        arrs = list(arrays)
        # arrs = [a.cpu().numpy() for a in arrs]
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
        self.model = self.model_cls(**self.model_params)
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


class TorchBaseStrategy(_TorchBaseStrategy, ABC):
    @abstractmethod
    def _build_criterion(
        self, criterion_cls: type[torch.nn.Module] | None, criterion_params: dict | None
    ) -> torch.nn.Module:
        pass

    @abstractmethod
    def _parse_sizes(self, *args, **kwargs) -> tuple[int, int]:
        """Parse input and output sizes from `train()` args and kwargs."""
        pass

    def _initialize_model(self, *args, **kwargs) -> torch.nn.Module:
        input_size, output_size = self._parse_sizes(*args, **kwargs)
        model_params = {**self.model_params, "input_size": input_size, "output_size": output_size}
        model = self.model_cls(**model_params)
        self.model = model
        self.model_params = model_params
        return model

    @abstractmethod
    def _build_data_loader(self, data: dict[str, torch.Tensor], *args, params: dict | None, **kwargs) -> Any:
        """Create data loader from `_preprocess()` output."""
        pass

    @abstractmethod
    def _epoch(
        self,
        data_loader,
        data: dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        *args,
        **kwargs,
    ) -> tuple[float, float]:
        """Training loop iteration. Should return train and test losses."""
        pass

    def _train(self, epochs: int, *args, **kwargs) -> tuple[list[float], list[float]]:
        """Main training loop."""
        pbar = tqdm(
            range(epochs), disable=log_config.disable_tqdm, desc=f"{TRAIN_LOSS_TEXT}: ...... | {TEST_LOSS_TEXT}: ......"
        )
        train_losses = []
        test_losses = []
        for _ in pbar:
            train_loss, test_loss = self._epoch(*args, **kwargs)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            pbar.set_description(f"{TRAIN_LOSS_TEXT}: {train_loss:.5f} | {TEST_LOSS_TEXT}: {test_loss:.5f}")
        return train_losses, test_losses

    def train(
        self,
        epochs: int,
        *args,
        optimizer_cls: type[torch.optim.Optimizer] | None = None,
        optimizer_params: dict | None = None,
        criterion_cls: type[torch.nn.Module] | None = None,
        criterion_params: dict | None = None,
        data_loader_params: dict | None = None,
        **kwargs,
    ) -> tuple[list[float], list[float]]:
        """Better be called from `super()` with kwargs when overriden."""
        self._initialize_model(*args, **kwargs).to(self.device)
        optimizer = self._build_optimizer(optimizer_cls, optimizer_params)
        criterion = self._build_criterion(criterion_cls, criterion_params)
        data = self._preprocess(*args, **kwargs)
        data_loader = self._build_data_loader(*args, data=data, params=data_loader_params)
        return self._train(
            *args, data_loader=data_loader, data=data, epochs=epochs, optimizer=optimizer, criterion=criterion, **kwargs
        )

    @abstractmethod
    def _predict(self, data: dict[str, torch.Tensor], *args, **kwargs) -> torch.Tensor:
        """Predict from _preprocess() data."""
        pass

    def predict(self, *args, **kwargs) -> np.ndarray:
        """Better be called from `super()` with kwargs when overriden."""
        model = self.model.to(self.device)
        model.eval()
        with torch.no_grad():
            data = self._preprocess(*args, **kwargs)
            pred = self._predict(data)
            return self._postprocess(pred)

    @abstractmethod
    def _preprocess(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        """Preprocess train or predict data to required formats."""
        pass

    @abstractmethod
    def _postprocess(self, t: torch.Tensor) -> np.ndarray:
        """Postprocess predict data after model prediction."""
        pass
