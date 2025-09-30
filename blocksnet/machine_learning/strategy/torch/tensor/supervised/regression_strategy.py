import torch
import numpy as np
from .base_strategy import TorchTensorSupervisedStrategy


class TorchTensorRegressionStrategy(TorchTensorSupervisedStrategy):
    """TorchTensorRegressionStrategy class.

    """
    def _build_criterion(
        self, criterion_cls: type[torch.nn.Module] | None = None, criterion_params: dict | None = None
    ):
        """Build criterion.

        Parameters
        ----------
        criterion_cls : type[torch.nn.Module] | None, default: None
            Description.
        criterion_params : dict | None, default: None
            Description.

        """
        criterion_cls = criterion_cls or torch.nn.MSELoss
        criterion_params = criterion_params or {}
        return criterion_cls(**criterion_params)

    def _x_to_tensor(self, x: np.ndarray) -> torch.Tensor:
        """X to tensor.

        Parameters
        ----------
        x : np.ndarray
            Description.

        Returns
        -------
        torch.Tensor
            Description.

        """
        return torch.tensor(x, dtype=torch.float32, device=self.device)

    def _y_to_tensor(self, y: np.ndarray) -> torch.Tensor:
        """Y to tensor.

        Parameters
        ----------
        y : np.ndarray
            Description.

        Returns
        -------
        torch.Tensor
            Description.

        """
        return torch.tensor(y, dtype=torch.float32, device=self.device)

    def _parse_sizes(self, x_train: np.ndarray, y_train: np.ndarray, **kwargs) -> tuple[int, int]:
        """Parse sizes.

        Parameters
        ----------
        x_train : np.ndarray
            Description.
        y_train : np.ndarray
            Description.
        **kwargs : dict
            Description.

        Returns
        -------
        tuple[int, int]
            Description.

        """
        return x_train.shape[1], y_train.shape[1]
