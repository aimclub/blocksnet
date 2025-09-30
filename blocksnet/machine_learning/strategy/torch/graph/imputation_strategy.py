import torch
import torch.nn as nn
import numpy as np
from abc import ABC
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from .base_strategy import TorchGraphBaseStrategy


class _TorchGraphImputationStrategy(TorchGraphBaseStrategy, ABC):
    """_TorchGraphImputationStrategy class.

    """
    def train(
        self,
        x: np.ndarray,
        y: np.ndarray,
        edge_index: np.ndarray,
        train_mask: np.ndarray,
        test_mask: np.ndarray,
        epochs: int = 100,
        optimizer_cls: type[torch.optim.Optimizer] | None = None,
        optimizer_params: dict | None = None,
        criterion_cls: type[torch.nn.Module] | None = None,
        criterion_params: dict | None = None,
        data_loader_params: dict | None = None,
    ) -> tuple[list[float], list[float]]:
        """Train.

        Parameters
        ----------
        x : np.ndarray
            Description.
        y : np.ndarray
            Description.
        edge_index : np.ndarray
            Description.
        train_mask : np.ndarray
            Description.
        test_mask : np.ndarray
            Description.
        epochs : int, default: 100
            Description.
        optimizer_cls : type[torch.optim.Optimizer] | None, default: None
            Description.
        optimizer_params : dict | None, default: None
            Description.
        criterion_cls : type[torch.nn.Module] | None, default: None
            Description.
        criterion_params : dict | None, default: None
            Description.
        data_loader_params : dict | None, default: None
            Description.

        Returns
        -------
        tuple[list[float], list[float]]
            Description.

        """
        return super().train(
            x=x,
            y=y,
            edge_index=edge_index,
            train_mask=train_mask,
            test_mask=test_mask,
            epochs=epochs,
            optimizer_cls=optimizer_cls,
            optimizer_params=optimizer_params,
            criterion_cls=criterion_cls,
            criterion_params=criterion_params,
            data_loader_params=data_loader_params,
        )

    def predict(self, x: np.ndarray, y: np.ndarray, edge_index: np.ndarray, imputation_mask: np.ndarray):
        """Predict.

        Parameters
        ----------
        x : np.ndarray
            Description.
        y : np.ndarray
            Description.
        edge_index : np.ndarray
            Description.
        imputation_mask : np.ndarray
            Description.

        """
        return super().predict(x=x, y=y, edge_index=edge_index, imputation_mask=imputation_mask)


class TorchGraphImputationStrategy(_TorchGraphImputationStrategy):
    """TorchGraphImputationStrategy class.

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

    def _parse_sizes(self, x: np.ndarray, y: np.ndarray, **kwargs) -> tuple[int, int]:
        """Parse sizes.

        Parameters
        ----------
        x : np.ndarray
            Description.
        y : np.ndarray
            Description.
        **kwargs : dict
            Description.

        Returns
        -------
        tuple[int, int]
            Description.

        """
        input_size = x.shape[1] + y.shape[1]
        output_size = y.shape[1]
        return input_size, output_size

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

    def _edge_index_to_tensor(self, edge_index: np.ndarray) -> torch.Tensor:
        """Edge index to tensor.

        Parameters
        ----------
        edge_index : np.ndarray
            Description.

        Returns
        -------
        torch.Tensor
            Description.

        """
        return torch.tensor(edge_index, dtype=torch.long, device=self.device)

    def _edge_attr_to_tensor(self, edge_attr: np.ndarray) -> torch.Tensor:
        """Edge attr to tensor.

        Parameters
        ----------
        edge_attr : np.ndarray
            Description.

        Returns
        -------
        torch.Tensor
            Description.

        """
        return torch.tensor(edge_attr, dtype=torch.float32, device=self.device)

    def _preprocess(
        self,
        x: np.ndarray,
        y: np.ndarray,
        edge_index: np.ndarray,
        edge_attr: np.ndarray | None = None,
        train_mask: np.ndarray | None = None,
        test_mask: np.ndarray | None = None,
        imputation_mask: np.ndarray | None = None,
    ) -> dict[str, torch.Tensor]:

        """Preprocess.

        Parameters
        ----------
        x : np.ndarray
            Description.
        y : np.ndarray
            Description.
        edge_index : np.ndarray
            Description.
        edge_attr : np.ndarray | None, default: None
            Description.
        train_mask : np.ndarray | None, default: None
            Description.
        test_mask : np.ndarray | None, default: None
            Description.
        imputation_mask : np.ndarray | None, default: None
            Description.

        Returns
        -------
        dict[str, torch.Tensor]
            Description.

        """
        result = {}
        is_fit_scaler = train_mask is not None

        (x,) = self._scale("x", is_fit_scaler, x)
        (y,) = self._scale("y", is_fit_scaler, y)
        result["x"] = self._x_to_tensor(x)
        result["y"] = self._y_to_tensor(y)
        result["edge_index"] = self._edge_index_to_tensor(edge_index)

        if edge_attr is not None:
            (edge_attr,) = self._scale("edge_attr", is_fit_scaler, edge_attr)
            result["edge_attr"] = self._edge_attr_to_tensor(edge_attr)

        if train_mask is not None:
            result["train_mask"] = torch.tensor(train_mask, dtype=torch.bool, device=self.device)
            result["test_mask"] = torch.tensor(test_mask, dtype=torch.bool, device=self.device)

        if imputation_mask is not None:
            result["imputation_mask"] = torch.tensor(imputation_mask, dtype=torch.bool, device=self.device)

        return result

    def _postprocess(self, y: torch.Tensor) -> np.ndarray:
        """Postprocess.

        Parameters
        ----------
        y : torch.Tensor
            Description.

        Returns
        -------
        np.ndarray
            Description.

        """
        a = y.cpu().numpy()
        (a,) = self._inverse_scale("y", a)
        return a

    def _epoch_train(self, batch: Data, optimizer: torch.optim.Optimizer, criterion: torch.nn.Module) -> float:
        """Epoch train.

        Parameters
        ----------
        batch : Data
            Description.
        optimizer : torch.optim.Optimizer
            Description.
        criterion : torch.nn.Module
            Description.

        Returns
        -------
        float
            Description.

        """
        model = self.model
        model.train()
        optimizer.zero_grad()

        imputation_mask = batch.imputation_mask
        y_masked = self._apply_imputation_mask(batch.y, imputation_mask)

        full_input = torch.cat([batch.x, y_masked], dim=1)

        y_pred = model(full_input, batch.edge_index, batch.edge_attr)

        train_mask = batch.train_mask
        total_mask = train_mask.unsqueeze(1) & imputation_mask
        loss = criterion(y_pred[total_mask], batch.y[total_mask])
        loss.backward()
        optimizer.step()
        return loss.item()

    def _epoch_test(
        self,
        batch: Data,
        criterion: torch.nn.Module,
    ) -> float:
        """Epoch test.

        Parameters
        ----------
        batch : Data
            Description.
        criterion : torch.nn.Module
            Description.

        Returns
        -------
        float
            Description.

        """
        model = self.model
        model.eval()

        imputation_mask = batch.imputation_mask
        y_masked = self._apply_imputation_mask(batch.y, imputation_mask)
        full_input = torch.cat([batch.x, y_masked], dim=1)

        with torch.no_grad():
            y_pred = model(full_input, batch.edge_index, batch.edge_attr)

        # Compute loss for missing values
        test_mask = batch.test_mask
        total_mask = test_mask.unsqueeze(1) & imputation_mask
        loss = criterion(y_pred[total_mask], batch.y[total_mask])
        return loss.item()

    def _epoch(
        self,
        data_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        *args,
        **kwargs,
    ) -> tuple[float, float]:
        """Epoch.

        Parameters
        ----------
        data_loader : DataLoader
            Description.
        optimizer : torch.optim.Optimizer
            Description.
        criterion : torch.nn.Module
            Description.
        *args : tuple
            Description.
        **kwargs : dict
            Description.

        Returns
        -------
        tuple[float, float]
            Description.

        """
        batch_train_losses = []
        batch_test_losses = []

        batch_list = list(data_loader)

        for batch in batch_list:
            batch.imputation_mask = self._create_imputation_mask(batch.y)

        for batch in batch_list:
            # train
            batch_train_loss = self._epoch_train(batch, optimizer, criterion)
            batch_train_losses.append(batch_train_loss)
            # test
            batch_test_loss = self._epoch_test(batch, criterion)
            batch_test_losses.append(batch_test_loss)

        train_loss = float(np.mean(batch_train_losses))
        test_loss = float(np.mean(batch_test_losses))
        return train_loss, test_loss

    def _predict(self, data: dict, *args, **kwargs) -> torch.Tensor:
        """Predict.

        Parameters
        ----------
        data : dict
            Description.
        *args : tuple
            Description.
        **kwargs : dict
            Description.

        Returns
        -------
        torch.Tensor
            Description.

        """
        x = data["x"]
        y = data["y"]
        edge_index = data["edge_index"]
        imputation_mask = data["imputation_mask"]

        dummy_targets = self._apply_imputation_mask(y, imputation_mask)
        full_input = torch.cat([x, dummy_targets], dim=1)
        y_pred_tensor = self.model(full_input, edge_index, data.get("edge_attr", None))
        return y_pred_tensor

    def _create_imputation_mask(self, targets: torch.Tensor, missing_rate: float = 0.4) -> torch.Tensor:
        """Create imputation mask.

        Parameters
        ----------
        targets : torch.Tensor
            Description.
        missing_rate : float, default: 0.4
            Description.

        Returns
        -------
        torch.Tensor
            Description.

        """
        n_nodes, n_targets = targets.shape
        imputation_mask = torch.zeros_like(targets, dtype=torch.bool, device=self.device)

        for i in range(n_targets):
            missing_nodes = torch.randperm(n_nodes)[: int(n_nodes * missing_rate)]
            imputation_mask[missing_nodes, i] = True

        return imputation_mask.to(self.device)

    def _apply_imputation_mask(self, targets: torch.Tensor, imputation_mask: torch.Tensor) -> torch.Tensor:
        """
        Apply imputation_mask to targets, setting missing values to 0.
        """
        masked_targets = targets.clone()
        masked_targets[imputation_mask] = 0
        return masked_targets
