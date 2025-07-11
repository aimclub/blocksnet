import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from .base_strategy import TorchGraphBaseStrategy

class TorchGraphImputationStrategy(TorchGraphBaseStrategy):
    def _build_criterion(
        self, criterion_cls: type[torch.nn.Module] | None = None, criterion_params: dict | None = None
    ):
        criterion_cls = criterion_cls or torch.nn.MSELoss
        criterion_params = criterion_params or {}
        return criterion_cls(**criterion_params)

    def _create_target_mask(self, targets: np.ndarray, missing_rate: float = 0.4) -> np.ndarray:
        """
        Create missing mask for target features.
        """
        mask = np.ones_like(targets, dtype=bool)
        n_nodes, n_targets = targets.shape
        for i in range(n_targets):
            missing_nodes = np.random.choice(n_nodes, int(n_nodes * missing_rate), replace=False)
            mask[missing_nodes, i] = False
        return mask

    def _apply_target_mask(self, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Apply mask to targets, setting missing values to 0.
        """
        masked_targets = targets.clone()
        masked_targets[~mask] = 0
        return masked_targets

    def _epoch_train(
        self, batch: Data, optimizer: torch.optim.Optimizer, criterion: torch.nn.Module
    ) -> float:
        """
        Train one epoch, computing loss only for missing target values.
        """
        model = self.model
        model.train()
        optimizer.zero_grad()

        # Get mask from batch (created in _preprocess)
        mask_tensor = batch.mask
        train_mask = batch.train_mask

        # Apply mask to targets
        masked_y = self._apply_target_mask(batch.y, mask_tensor)

        # Create full input with masked targets
        full_input = torch.cat([batch.x, masked_y], dim=1)

        # Forward pass
        y_pred = model(full_input, batch.edge_index)

        # Compute loss only for missing values (where mask is False)
        total_mask = train_mask.unsqueeze(1) & ~mask_tensor
        if total_mask.sum() > 0:
            loss = criterion(y_pred[total_mask], batch.y[total_mask])
        else:
            loss = torch.tensor(0.0, device=self.device)

        loss.backward()
        optimizer.step()
        return loss.item()

    def _epoch_test(
        self, x_test: torch.Tensor, y_test: torch.Tensor, edge_index: torch.Tensor, 
        criterion: torch.nn.Module, val_mask: torch.Tensor, mask: torch.Tensor
    ) -> float:
        """
        Validate model, computing loss only for missing target values.
        Returns: float (loss value)
        """
        model = self.model
        model.eval()
        with torch.no_grad():
            # Apply mask to test targets
            masked_y = self._apply_target_mask(y_test, mask)

            # Create full input
            full_input = torch.cat([x_test, masked_y], dim=1)

            # Forward pass
            y_pred = model(full_input, edge_index)

            # Compute loss for missing values
            total_mask = val_mask.unsqueeze(1) & ~mask
            if total_mask.sum() > 0:
                loss = criterion(y_pred[total_mask], y_test[total_mask])
                return loss.item()
            else:
                return 0.0

    def _epoch(
        self,
        data_loader: DataLoader,
        data: dict,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        **kwargs,
    ) -> tuple[float, dict]:
        """
        Run one epoch, returning train loss and validation metrics.
        """
        batch_losses = []
        for batch in data_loader:
            batch_loss = self._epoch_train(batch, optimizer, criterion)
            batch_losses.append(batch_loss)
        train_loss = float(np.mean(batch_losses))
        test_metrics = self._epoch_test(
            data["x_test"], data["y_test"], data["edge_index"], 
            criterion, data["val_mask"], data["mask"]
        )
        return train_loss, test_metrics

    def _parse_sizes(self, x_train: np.ndarray, y_train: np.ndarray, **kwargs) -> tuple[int, int]:
        input_size = x_train.shape[1] + y_train.shape[1]
        output_size = y_train.shape[1]
        return input_size, output_size

    def _x_to_tensor(self, x: np.ndarray) -> torch.Tensor:
        return torch.tensor(x, dtype=torch.float32, device=self.device)

    def _y_to_tensor(self, y: np.ndarray) -> torch.Tensor:
        return torch.tensor(y, dtype=torch.float32, device=self.device)

    def _preprocess(
        self,
        x: np.ndarray | None = None,
        y: np.ndarray | None = None,
        x_train: np.ndarray | None = None,
        x_test: np.ndarray | None = None,
        y_train: np.ndarray | None = None,
        y_test: np.ndarray | None = None,
        edge_index: torch.Tensor | None = None,
        train_indices: np.ndarray | None = None,
        val_indices: np.ndarray | None = None,
        **kwargs,
    ) -> dict:
        result = {}

        # Scaling and transforming features
        if x is not None:
            (x,) = self._scale("x", False, x)
            result["x"] = self._x_to_tensor(x)
        if y is not None:
            (y,) = self._scale("y", False, y)
            result["y"] = self._y_to_tensor(y)

        if x_train is not None and x_test is not None:
            x_train, x_test = self._scale("x", True, x_train, x_test)
            result["x_train"] = self._x_to_tensor(x_train)
            result["x_test"] = self._x_to_tensor(x_test)
        if y_train is not None and y_test is not None:
            y_train, y_test = self._scale("y", True, y_train, y_test)
            result["y_train"] = self._y_to_tensor(y_train)
            result["y_test"] = self._y_to_tensor(y_test)

        if edge_index is not None:
            result["edge_index"] = edge_index.to(self.device)

        # Create train/val masks
        if train_indices is not None and val_indices is not None:
            num_nodes = result["x_train"].shape[0]
            train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=self.device)
            val_mask = torch.zeros(num_nodes, dtype=torch.bool, device=self.device)
            train_mask[train_indices] = True
            val_mask[val_indices] = True
            result["train_mask"] = train_mask
            result["val_mask"] = val_mask

        # Create target mask for missing values
        if y_train is not None:
            mask = self._create_target_mask(result["y_train"].cpu().numpy())
            result["mask"] = torch.tensor(mask, dtype=torch.bool, device=self.device)

        return result

    def _postprocess(self, y: torch.Tensor) -> np.ndarray:
        a = y.cpu().numpy()
        (a,) = self._inverse_scale("y", a)
        return a

    def train(
        self,
        x: np.ndarray,
        y: np.ndarray,
        edge_index: torch.Tensor,
        train_indices: np.ndarray,
        val_indices: np.ndarray,
        epochs: int = 100,
        batch_size: int = 1,  # Set to 1 to match old code (whole graph)
        optimizer_cls: type[torch.optim.Optimizer] | None = None,
        optimizer_params: dict | None = None,
        criterion_cls: type[torch.nn.Module] | None = None,
        criterion_params: dict | None = None,
        data_loader_params: dict | None = None,
        **kwargs,
    ) -> tuple[list[float], list[dict]]:
        self._initialize_model(x, y)
        optimizer = self._build_optimizer(optimizer_cls, optimizer_params)
        criterion = self._build_criterion(criterion_cls, criterion_params)
        data = self._preprocess(
            x_train=x, x_test=x, y_train=y, y_test=y, edge_index=edge_index,
            train_indices=train_indices, val_indices=val_indices
        )
        data_loader = self._build_data_loader(data, data_loader_params or {"batch_size": batch_size, "shuffle": False})
        
        train_losses = []
        val_metrics = []
        for epoch in range(epochs):
            train_loss, test_metrics = self._epoch(
                data_loader, data, optimizer, criterion
            )
            train_losses.append(train_loss)
            val_metrics.append(test_metrics)
        
        return train_losses, val_metrics

    def _build_data_loader(self, data: dict, params: dict | None) -> DataLoader:
        params = params or {"batch_size": 1, "shuffle": False}
        data_entry = Data(
            x=data["x_train"], 
            y=data["y_train"], 
            edge_index=data["edge_index"],
            train_mask=data.get("train_mask"), 
            val_mask=data.get("val_mask"),
            mask=data.get("mask")  # Include target mask
        )
        return DataLoader([data_entry], **params)

    def _predict(self, data: dict, *args, **kwargs) -> torch.Tensor:
        x = data["x"]
        edge_index = data["edge_index"]
        mask = data.get("mask", torch.ones_like(data["y"], dtype=torch.bool, device=self.device))
        dummy_targets = self._apply_target_mask(data["y"], mask)
        full_input = torch.cat([x, dummy_targets], dim=1)
        y_pred_tensor = self.model(full_input, edge_index)
        return y_pred_tensor

    def predict(self, x: np.ndarray, edge_index: torch.Tensor, y: np.ndarray | None = None) -> np.ndarray:
        self.model.eval()
        data = self._preprocess(x=x, y=y, edge_index=edge_index)
        with torch.no_grad():
            y_pred_tensor = self._predict(data)
            y_pred = self._postprocess(y_pred_tensor)
        return y_pred