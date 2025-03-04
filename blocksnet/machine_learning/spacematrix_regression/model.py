import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from tqdm import tqdm
from loguru import logger
from ...common.config import log_config


class Model(torch.nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_layer_size: int = 8):
        super(Model, self).__init__()
        self.conv1 = SAGEConv(input_size, hidden_layer_size)
        self.conv2 = SAGEConv(hidden_layer_size, output_size)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        gsi = torch.sigmoid(x[:, 1])
        mxi = torch.sigmoid(x[:, 2])
        fsi = torch.relu(x[:, 0]) + gsi
        return torch.stack([fsi, gsi, mxi], dim=1)


def train_model(
    x,
    edge_index,
    y,
    train_mask,
    epochs: int = 1_000,
    learning_rate: float = 3e-4,
    weight_decay: float = 5e-4,
    model=None,
) -> tuple[Model, list[float]]:
    if model is None:
        logger.warning("No model is provided. Initializing.")
        input_size = x.shape[1]
        output_size = y.shape[1]
        model = Model(input_size, output_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    losses = []
    model.train()
    pbar = tqdm(range(epochs), disable=log_config.disable_tqdm, desc="Current loss : ....")
    for _ in pbar:
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = F.mse_loss(out[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        pbar.set_description(f"Current loss : {loss.item():.3f}")
    return model, losses


def eval_model(model, x, edge_index) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        out = model(x, edge_index)
    return out


def test_model(model, x, edge_index, y, test_mask) -> tuple[float, torch.Tensor]:
    out = eval_model(model, x, edge_index)
    loss = F.mse_loss(out[test_mask], y[test_mask])
    return loss.item(), out


def plot_losses(losses: list[float]):

    try:
        from matplotlib import pyplot as plt
    except ImportError:
        raise ImportError("MatPlotLib package is required but not installed.")

    plt.figure(figsize=(10, 5))
    plt.plot(losses[0:], label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training Loss over Epochs")
    plt.show()


__all__ = [
    "Model",
    "train_model",
    "test_model",
    "eval_model",
    "plot_losses",
]
