import torch
from abc import ABC, abstractmethod
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from ..base_strategy import TorchBaseStrategy


class TorchGraphBaseStrategy(TorchBaseStrategy, ABC):
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
        data_entry = Data(**data)
        return DataLoader([data_entry], **params)
