import torch
import numpy as np
from .base_strategy import TorchGraphSupervisedStrategy


class TorchGraphRegressionStrategy(TorchGraphSupervisedStrategy):
    def _build_criterion(
        self, criterion_cls: type[torch.nn.Module] | None = None, criterion_params: dict | None = None
    ):
        criterion_cls = criterion_cls or torch.nn.MSELoss
        criterion_params = criterion_params or {}
        return criterion_cls(**criterion_params)

    def _x_to_tensor(self, x: np.ndarray) -> torch.Tensor:
        return torch.tensor(x, dtype=torch.float32, device=self.device)

    def _y_to_tensor(self, y: np.ndarray) -> torch.Tensor:
        return torch.tensor(y, dtype=torch.float32, device=self.device)

    def _edge_attr_to_tensor(self, edge_attr: np.ndarray) -> torch.Tensor:
        return torch.tensor(edge_attr, dtype=torch.float32, device=self.device)

    def _parse_sizes(self, x: np.ndarray, y: np.ndarray, **kwargs) -> tuple[int, int]:
        return x.shape[1], y.shape[1]
