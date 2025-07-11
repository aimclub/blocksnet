import torch
import numpy as np
from .base_strategy import TorchTensorBaseClassificationStrategy


class TorchTensorMultiLabelClassificationStrategy(TorchTensorBaseClassificationStrategy):

    _default_criterion_cls = torch.nn.BCEWithLogitsLoss

    def _y_to_tensor(self, y: np.ndarray) -> torch.Tensor:
        return torch.tensor(y, dtype=torch.float32, device=self.device)

    def _parse_sizes(self, x_train: np.ndarray, y_train: np.ndarray, **kwargs) -> tuple[int, int]:
        input_size = x_train.shape[1]
        num_classes = y_train.shape[1]
        return input_size, num_classes
