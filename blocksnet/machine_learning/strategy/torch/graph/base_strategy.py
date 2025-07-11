import torch
from abc import ABC
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from ..base_strategy import TorchBaseStrategy


class TorchGraphBaseStrategy(TorchBaseStrategy, ABC):
    def _build_data_loader(self, data: dict[str, torch.Tensor], params: dict | None) -> DataLoader:
        params = params or {"batch_size": 32, "shuffle": True}
        data_entry = Data(**data)
        return DataLoader([data_entry], **params)
