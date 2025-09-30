import torch
import numpy as np
from .base_strategy import TorchTensorBaseClassificationStrategy


class TorchTensorMulticlassClassificationStrategy(TorchTensorBaseClassificationStrategy):

    """TorchTensorMulticlassClassificationStrategy class.

    """
    _default_criterion_cls = torch.nn.CrossEntropyLoss

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
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.squeeze(-1)
        return torch.tensor(y, dtype=torch.long, device=self.device)

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
        input_size = x_train.shape[1]
        num_classes = int(np.max(y_train)) + 1
        return input_size, num_classes
