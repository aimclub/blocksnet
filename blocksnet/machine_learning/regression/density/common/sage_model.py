import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, Linear
from tqdm import tqdm
from loguru import logger
from .....config import log_config


class SageModel(torch.nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_channels=[32, 64, 128]):
        super(SageModel, self).__init__()
        
        # Энкодер (сжимающая часть)
        self.enc_fc = Linear(input_size, hidden_channels[0])
        self.enc_conv1 = SAGEConv(hidden_channels[0], hidden_channels[1])
        self.enc_conv2 = SAGEConv(hidden_channels[1], hidden_channels[2])
        
        # Боттлнек
        self.bottleneck = SAGEConv(hidden_channels[2], hidden_channels[2])
        
        # Декодер (расширяющая часть)
        self.dec_conv1 = SAGEConv(hidden_channels[2]*2, hidden_channels[1])  # *2 из-за конкатенации
        self.dec_conv2 = SAGEConv(hidden_channels[1]*2, hidden_channels[0])  # *2 из-за конкатенации
        
        # Исправленный размер для dec_fc с учетом конкатенации skip2
        self.dec_fc = Linear(hidden_channels[0]*2, output_size)  # *2 из-за последней конкатенации
        
        # Дополнительные преобразования для согласования размеров в skip connections
        self.skip_transform1 = Linear(hidden_channels[1], hidden_channels[1])
        self.skip_transform2 = Linear(hidden_channels[0], hidden_channels[0])

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        
        x0 = F.tanh(self.enc_fc(x))
        x1 = F.tanh(self.enc_conv1(x0, edge_index))
        x2 = F.tanh(self.enc_conv2(x1, edge_index))
        
        x3 = F.tanh(self.bottleneck(x2, edge_index))
        
        skip1 = self.skip_transform1(x1) 
        x = torch.cat([x3, x2], dim=1)
        x = F.tanh(self.dec_conv1(x, edge_index))
        
        skip2 = self.skip_transform2(x0)
        x = torch.cat([x, skip1], dim=1)
        x = F.tanh(self.dec_conv2(x, edge_index))
        
        x = torch.cat([x, skip2], dim=1)
        x = self.dec_fc(x)        

        gsi = x[:, 1]
        mxi = x[:, 2]
        fsi = x[:, 0]         
        
        return torch.stack([fsi, gsi, mxi], dim=1)
