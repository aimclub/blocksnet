import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data

class ServiceGNN(nn.Module):
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int = 64, 
                 output_dim: int = 57) -> None:
        """
        Инициализация графовой нейронной сети для анализа сервисов

        Params:
            input_dim -- Размерность входных признаков узлов графа
            hidden_dim -- Размерность скрытых слоев графовых сверток
            output_dim -- Размерность выходных данных (например, количество сервисов)

        Returns:
            None
        """
        super().__init__()
        self.conv1 = GATConv(input_dim, hidden_dim)
        self.conv2 = GATConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, data: Data) -> torch.Tensor:
        """
        Прямой проход через графовую нейронную сеть

        Params:
            data -- Объект данных графа

        Returns:
            Выходной тензор, представляющий предсказания для каждого узла графа.
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        return self.fc(x)