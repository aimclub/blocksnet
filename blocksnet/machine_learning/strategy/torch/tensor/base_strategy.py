import torch
from abc import ABC, abstractmethod
from torch.utils.data import TensorDataset, DataLoader
from ..base_strategy import TorchBaseStrategy


class TorchTensorBaseStrategy(TorchBaseStrategy, ABC):
    @abstractmethod
    """TorchTensorBaseStrategy class.

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
        tensors = [data[k] for k in ["x_train", "y_train"] if k in data]
        dataset = TensorDataset(*tensors)
        return DataLoader(dataset, **params)
