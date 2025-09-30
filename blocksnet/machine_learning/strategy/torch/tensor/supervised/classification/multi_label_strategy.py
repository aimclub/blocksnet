import torch
import numpy as np
from .base_strategy import TorchTensorBaseClassificationStrategy


class TorchTensorMultiLabelClassificationStrategy(TorchTensorBaseClassificationStrategy):

    """TorchTensorMultiLabelClassificationStrategy class.

    """
    _default_criterion_cls = torch.nn.BCEWithLogitsLoss

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
        input_size = x_train.shape[1]
        num_classes = y_train.shape[1]
        return input_size, num_classes
