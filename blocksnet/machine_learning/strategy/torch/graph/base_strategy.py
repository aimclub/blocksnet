import torch
from abc import ABC, abstractmethod
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from ..base_strategy import TorchBaseStrategy


class TorchGraphBaseStrategy(TorchBaseStrategy, ABC):
    @abstractmethod
    """TorchGraphBaseStrategy class.

    """
    def _epoch(
        self,
        data_loader: DataLoader,
        data: dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
    ) -> tuple[float, float]:
        """Epoch.

        Parameters
        ----------
        data_loader : DataLoader
            Description.
        data : dict[str, torch.Tensor]
            Description.
        optimizer : torch.optim.Optimizer
            Description.
        criterion : torch.nn.Module
            Description.

        Returns
        -------
        tuple[float, float]
            Description.

        """
        pass

    def _build_data_loader(self, data: dict[str, torch.Tensor], params: dict | None) -> DataLoader:
        """Build data loader.

        Parameters
        ----------
        data : dict[str, torch.Tensor]
            Description.
        params : dict | None
            Description.

        Returns
        -------
        DataLoader
            Description.

        """
        params = params or {"batch_size": 32, "shuffle": True}
        data_entry = Data(**data)
        return DataLoader([data_entry], **params)
