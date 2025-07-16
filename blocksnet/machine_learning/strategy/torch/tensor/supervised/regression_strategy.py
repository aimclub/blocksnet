import torch
import numpy as np
from .base_strategy import TorchTensorSupervisedStrategy


class TorchTensorRegressionStrategy(TorchTensorSupervisedStrategy):
    def _build_criterion(
        self, criterion_cls: type[torch.nn.Module] | None = None, criterion_params: dict | None = None
    ):
        criterion_cls = criterion_cls or torch.nn.MSELoss
        criterion_params = criterion_params or {}
        return criterion_cls(**criterion_params)

    def _x_to_tensor(self, x: np.ndarray) -> torch.Tensor:
        return torch.tensor(x, dtype=torch.float32, device=self.device)

    def _y_to_tensor(self, x: np.ndarray) -> torch.Tensor:
        return torch.tensor(x, dtype=torch.float32, device=self.device)

    def _parse_sizes(self, x_train: np.ndarray, y_train: np.ndarray, **kwargs) -> tuple[int, int]:
        return x_train.shape[1], y_train.shape[1]
