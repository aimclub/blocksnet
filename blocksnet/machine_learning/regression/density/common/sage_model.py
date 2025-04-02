import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, Linear
from tqdm import tqdm
from loguru import logger
from .....config import log_config


class SageModel(torch.nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super(SageModel, self).__init__()
        self.fc1 = Linear(input_size, 16)
        self.conv1 = SAGEConv(16, 32)
        self.conv2 = SAGEConv(32, 16)
        self.fc2 = Linear(16, output_size)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        x = F.relu(self.fc1(x))
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.fc2(x)
        gsi = torch.sigmoid(x[:, 1])
        mxi = torch.sigmoid(x[:, 2])
        fsi = torch.relu(x[:, 0]) + gsi
        return torch.stack([fsi, gsi, mxi], dim=1)
