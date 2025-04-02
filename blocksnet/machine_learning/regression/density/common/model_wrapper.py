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

    def load_model(self, file_path: str):
        state_dict = torch.load(file_path, weights_only=True)
        self.model.load_state_dict(state_dict)

    def save_model(self, file_path: str):
        state_dict = self.model.state_dict()
        torch.save(state_dict, file_path)

    def _train_model(self, data: Data, epochs: int, learning_rate: float, weight_decay: float):
        model = self.model
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        pbar = tqdm(range(epochs), disable=log_config.disable_tqdm, desc="Current loss : ......")
        losses = []
        model.train()
        for _ in pbar:
            optimizer.zero_grad()
            out = model(data)
            loss = F.huber_loss(out[data.train_mask], data.y[data.train_mask], reduction="mean", delta=0.05)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            pbar.set_description(f"Current loss : {loss.item():.5f}")
        return losses

    def _evaluate_model(self, data: Data):
        model = self.model
        model.eval()
        with torch.no_grad():
            out = model(data)
        return out

    def _test_model(self, data: Data):
        out = self._evaluate_model(data)
        loss = F.huber_loss(out[data.test_mask], data.y[data.test_mask], reduction="mean", delta=0.05)
        return loss.item()
