import torch
from abc import ABC, abstractmethod
from torch.utils.data import TensorDataset, DataLoader
from ..base_strategy import TorchBaseStrategy


class TorchTensorBaseStrategy(TorchBaseStrategy, ABC):
    @abstractmethod
    def _epoch(
        self,
        data_loader: DataLoader,
        data: dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
    ) -> tuple[float, float]:
        pass

    def _build_data_loader(self, data: dict[str, torch.Tensor], params: dict | None) -> DataLoader:
        params = params or {"batch_size": 32, "shuffle": True}
        tensors = [data[k] for k in ["x_train", "y_train"] if k in data]
        dataset = TensorDataset(*tensors)
        return DataLoader(dataset, **params)
