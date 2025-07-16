from pathlib import Path
import torch
import torch.nn as nn
from blocksnet.machine_learning.strategy.torch import TorchGraphImputationStrategy
from torch_geometric.nn import GraphSAGE

CURRENT_DIRECTORY = Path(__file__).parent
ARTIFACTS_DIRECTORY = str(CURRENT_DIRECTORY / "artifacts")


class MultiScaleGNN(nn.Module):
    def __init__(
        self, input_size: int, output_size: int, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.2
    ):
        super().__init__()
        self.output_size = output_size
        self.graphsage = GraphSAGE(
            in_channels=input_size,
            hidden_channels=hidden_dim,
            num_layers=num_layers,
            out_channels=hidden_dim,
            dropout=dropout,
            act="relu",
            norm=nn.LayerNorm(hidden_dim),
            jk="cat",
        )
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, output_size),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        features = self.graphsage(x, edge_index)
        output = self.output_layer(features)
        return output


def get_default_strategy() -> TorchGraphImputationStrategy:
    strategy = TorchGraphImputationStrategy(model_cls=MultiScaleGNN)
    strategy.load(ARTIFACTS_DIRECTORY)
    return strategy
