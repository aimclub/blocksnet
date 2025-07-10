import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from blocksnet.config import log_config
from ..base_strategy import TorchBaseStrategy
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from typing import Tuple, List

TRAIN_LOSS_TEXT = "Train loss"
TEST_LOSS_TEXT = "Test loss"
TRAIN_R2_TEXT = "Train R²"
TEST_R2_TEXT = "Test R²"

class TorchGraphImputationStrategy(TorchBaseStrategy):
    def __init__(self, model_cls: type[torch.nn.Module] = None, scalers: dict | None = None):
        super().__init__(model_cls, scalers=scalers)

    def _build_optimizer(
        self, optimizer_cls: type[torch.optim.Optimizer] | None = None, optimizer_params: dict | None = None
    ):
        optimizer_cls = optimizer_cls or torch.optim.Adam
        optimizer_params = optimizer_params or {"lr": 1e-3, "weight_decay": 1e-4}
        return super()._build_optimizer(optimizer_cls, optimizer_params)

    def _build_criterion(
        self, criterion_cls: type[torch.nn.Module] | None = None, criterion_params: dict | None = None
    ):
        criterion_cls = criterion_cls or torch.nn.MSELoss
        criterion_params = criterion_params or {}
        return super()._build_criterion(criterion_cls, criterion_params)

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
        Apply mask to targets for imputation.
        """
        masked_targets = targets.clone()
        masked_targets[~mask] = 0
        return masked_targets

    def _create_dataloader(self, x: torch.Tensor, y: torch.Tensor, edge_index: torch.Tensor, 
                          batch_size: int, use_subgraphs: bool) -> DataLoader:
        """
        Create DataLoader for graph data using torch_geometric.
        """
        # Filter edge_index to include only valid nodes
        num_nodes = x.shape[0]
        valid_mask = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)
        filtered_edge_index = edge_index[:, valid_mask]
        
        # Create Data object
        data = Data(x=x, y=y, edge_index=filtered_edge_index)
        
        if not use_subgraphs:
            # Single graph: use batch_size equal to number of nodes
            return DataLoader([data], batch_size=num_nodes, shuffle=False)
        else:
            # Subgraphs: use specified batch_size
            return DataLoader([data], batch_size=batch_size, shuffle=True)

    def _create_dataloader(self, x: torch.Tensor, y: torch.Tensor, edge_index: torch.Tensor, 
                          batch_size: int, use_subgraphs: bool) -> DataLoader:
        """
        Create DataLoader for graph data using torch_geometric.
        """
        # Filter edge_index to include only valid nodes
        num_nodes = x.shape[0]
        valid_mask = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)
        filtered_edge_index = edge_index[:, valid_mask]
        
        # Create Data object
        data = Data(x=x, y=y, edge_index=filtered_edge_index)
        
        if not use_subgraphs:
            # Single graph: use batch_size equal to number of nodes
            return DataLoader([data], batch_size=num_nodes, shuffle=False)
        else:
            # Subgraphs: use specified batch_size
            return DataLoader([data], batch_size=batch_size, shuffle=True)

    def _epoch_train(self, batch: Data, optimizer: torch.optim.Optimizer, criterion: torch.nn.Module) -> Tuple[float, float]:
        """
        Train one epoch with masked targets for imputation, returning loss and R².
        """
        self._model.train()
        optimizer.zero_grad()
        
        # Create and apply mask to targets
        mask = self._create_target_mask(batch.y.cpu().numpy(), missing_rate=0.2)
        mask_tensor = torch.tensor(mask, dtype=torch.bool).to(self.device)
        masked_y = self._apply_target_mask(batch.y, mask_tensor)
        
        # Combine features and masked targets
        full_input = torch.cat([batch.x, masked_y], dim=1)
        pred = self._model(full_input, batch.edge_index)
        
        # Compute loss and R² only on missing values
        missing_mask = ~mask_tensor
        if missing_mask.sum() > 0:
            loss = criterion(pred[missing_mask], batch.y[missing_mask])
            y_true = batch.y[missing_mask]
            y_pred = pred[missing_mask]
            ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
            ss_res = torch.sum((y_true - y_pred) ** 2)
            r2 = 1 - ss_res / (ss_tot + 1e-8)  # Add small epsilon to avoid division by zero
        else:
            loss = torch.tensor(0.0, device=self.device)
            r2 = torch.tensor(0.0, device=self.device)
        
        loss.backward()
        optimizer.step()
        return loss.item(), r2.item()

    def _epoch_test(self, x_test: torch.Tensor, y_test: torch.Tensor, edge_index: torch.Tensor, 
                    criterion: torch.nn.Module) -> Tuple[float, float]:
        """
        Test one epoch with masked targets for imputation, returning loss and R².
        """
        self._model.eval()
        with torch.no_grad():
            # Filter edge_index for test data
            num_nodes = x_test.shape[0]
            valid_mask = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)
            filtered_edge_index = edge_index[:, valid_mask]
            
            # Create and apply mask to targets
            mask = self._create_target_mask(y_test.cpu().numpy(), missing_rate=0.4)
            mask_tensor = torch.tensor(mask, dtype=torch.bool).to(self.device)
            masked_y_test = self._apply_target_mask(y_test, mask_tensor)
            
            # Combine features and masked targets
            full_input = torch.cat([x_test, masked_y_test], dim=1)
            pred = self._model(full_input, filtered_edge_index)
            
            # Compute loss and R² only on missing values
            missing_mask = ~mask_tensor
            if missing_mask.sum() > 0:
                loss = criterion(pred[missing_mask], y_test[missing_mask])
                y_true = y_test[missing_mask]
                y_pred = pred[missing_mask]
                ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
                ss_res = torch.sum((y_true - y_pred) ** 2)
                r2 = 1 - ss_res / (ss_tot + 1e-8)  # Add small epsilon to avoid division by zero
            else:
                loss = torch.tensor(0.0, device=self.device)
                r2 = torch.tensor(0.0, device=self.device)
        return loss.item(), r2.item()

    def train(
        self,
        x: np.ndarray,
        y: np.ndarray,
        edge_index: torch.Tensor,
        train_indices: np.ndarray,
        val_indices: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        optimizer_cls: type[torch.optim.Optimizer] | None = None,
        optimizer_params: dict | None = None,
        criterion_cls: type[torch.nn.Module] | None = None,
        criterion_params: dict | None = None,
        use_subgraphs: bool = False
    ) -> Tuple[List[float], List[float], List[float], List[float], torch.Tensor]:
        """
        Full training loop with train and validation indices, including R² metric.
        """
        # Split data using indices
        x_train = x[train_indices]
        y_train = y[train_indices]
        x_test = x[val_indices]
        y_test = y[val_indices]
        
        # Initialize model
        input_dim = x_train.shape[1] + y_train.shape[1]
        output_dim = y_train.shape[1]
        self._model = self._build_model(input_dim, output_dim)
        
        # Build optimizer and criterion
        optimizer = self._build_optimizer(optimizer_cls, optimizer_params)
        criterion = self._build_criterion(criterion_cls, criterion_params)
        
        # Preprocess data
        x_train_tensor, x_test_tensor = self._preprocess("x", True, x_train, x_test)
        y_train_tensor, y_test_tensor = self._preprocess("y", True, y_train, y_test)
        
        # Create DataLoader
        dataloader = self._create_dataloader(x_train_tensor, y_train_tensor, edge_index, 
                                           batch_size, use_subgraphs)
        
        # Training loop
        train_losses = []
        test_losses = []
        train_r2_scores = []
        test_r2_scores = []
        pbar = tqdm(
            range(epochs), disable=log_config.disable_tqdm, 
            desc=f"{TRAIN_LOSS_TEXT}: ...... | {TEST_LOSS_TEXT}: ...... | {TRAIN_R2_TEXT}: ...... | {TEST_R2_TEXT}: ......"
        )
        
        for _ in pbar:
            batch_losses = []
            batch_r2_scores = []
            for batch in dataloader:
                batch_loss, batch_r2 = self._epoch_train(batch, optimizer, criterion)
                batch_losses.append(batch_loss)
                batch_r2_scores.append(batch_r2)
            
            train_loss = np.mean(batch_losses)
            train_r2 = np.mean(batch_r2_scores)
            test_loss, test_r2 = self._epoch_test(x_test_tensor, y_test_tensor, edge_index, criterion)
            
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            train_r2_scores.append(train_r2)
            test_r2_scores.append(test_r2)
            
            pbar.set_description(
                f"{TRAIN_LOSS_TEXT}: {train_loss:.5f} | {TEST_LOSS_TEXT}: {test_loss:.5f} | "
                f"{TRAIN_R2_TEXT}: {train_r2:.5f} | {TEST_R2_TEXT}: {test_r2:.5f}"
            )
        
        return train_losses, test_losses, train_r2_scores, test_r2_scores

    def predict(self, x: np.ndarray, edge_index: torch.Tensor) -> np.ndarray:
        """
        Predict target values.
        """
        self._model.eval()
        (x_tensor,) = self._preprocess("x", False, x)
        with torch.no_grad():
            # Filter edge_index for prediction
            num_nodes = x.shape[0]
            valid_mask = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)
            filtered_edge_index = edge_index[:, valid_mask]
            
            # Use zeroed-out targets for prediction (simulating full imputation)
            dummy_targets = torch.zeros((x.shape[0], self._model.output_size), device=self.device)
            full_input = torch.cat([x_tensor, dummy_targets], dim=1)
            y_pred_tensor = self._model(full_input, filtered_edge_index)
            (y_pred,) = self._postprocess("y", y_pred_tensor)
        return y_pred