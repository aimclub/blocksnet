import torch
from torch_geometric.data import Data
from tqdm import tqdm
from .....config import log_config
import torch.nn.functional as F


class ModelWrapper:
    def __init__(self, n_features, n_targets, model_class: type[torch.nn.Module], *args, **kwargs):
        self.n_features = n_features
        self.n_targets = n_targets
        self.model = model_class(n_features, n_targets, *args, **kwargs)
                # Определяем функцию потерь в зависимости от параметра loss_fn
        self.loss_functions = {
            "mse": lambda pred, target: F.mse_loss(pred, target, reduction="mean"),
            "mae": lambda pred, target: F.l1_loss(pred, target, reduction="mean"),
            "huber": lambda pred, target: F.huber_loss(pred, target, reduction="mean", delta=0.05)
        }

    def load_model(self, file_path: str):
        state_dict = torch.load(file_path, weights_only=True)
        self.model.load_state_dict(state_dict)

    def save_model(self, file_path: str):
        state_dict = self.model.state_dict()
        torch.save(state_dict, file_path)

    def _train_model(self, data: Data, epochs: int, learning_rate: float, weight_decay: float, loss_fn: str = "huber"):
        model = self.model
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        pbar = tqdm(range(epochs), disable=log_config.disable_tqdm, desc="Train loss: ...... | Val loss: ......")
        
        train_losses = []
        val_losses = []
    
        if loss_fn not in self.loss_functions:
            raise ValueError(f"Unsupported loss function: {loss_fn}. Choose from 'mse', 'mae', or 'huber'.")
        
        selected_loss_fn = self.loss_functions[loss_fn]
        
        model.train()
        for _ in pbar:
            # Training phase
            optimizer.zero_grad()
            out = model(data)
            train_loss = selected_loss_fn(out[data.train_mask], data.y[data.train_mask])
            train_loss.backward()
            optimizer.step()
            train_losses.append(train_loss.item())
            
            # Validation phase
            model.eval()
            with torch.no_grad():
                val_out = model(data)
                val_loss = selected_loss_fn(val_out[data.test_mask], data.y[data.test_mask])  # Исправлено на val_mask
            val_losses.append(val_loss.item())
            
            model.train()  # Switch back to training mode
            
            pbar.set_description(f"Train loss: {train_loss.item():.5f} | Val loss: {val_loss.item():.5f}")
        
        return train_losses, val_losses

    def _evaluate_model(self, data: Data):
        model = self.model
        model.eval()
        with torch.no_grad():
            out = model(data)
        return out

    def _test_model(self, data: Data, loss_fn: str = "huber"):
        out = self._evaluate_model(data)
    
        
        if loss_fn not in self.loss_functions:
            raise ValueError(f"Unsupported loss function: {loss_fn}. Choose from 'mse', 'mae', or 'huber'.")
        
        selected_loss_fn = self.loss_functions[loss_fn]
        loss = selected_loss_fn(out[data.test_mask], data.y[data.test_mask])
        return loss.item()