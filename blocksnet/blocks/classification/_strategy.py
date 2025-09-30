from pathlib import Path
import torch
import torch.nn as nn
from blocksnet.machine_learning.strategy.torch import TorchTensorMulticlassClassificationStrategy

CURRENT_DIRECTORY = Path(__file__).parent
ARTIFACTS_DIRECTORY = str(CURRENT_DIRECTORY / "artifacts")


class MulticlassClassifier(nn.Module):
    """MulticlassClassifier class.

    """
    def __init__(self, input_size: int, output_size: int):
        """Initialize the instance.

        Parameters
        ----------
        input_size : int
            Description.
        output_size : int
            Description.

        Returns
        -------
        None
            Description.

        """
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        self.head = nn.Linear(32, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward.

        Parameters
        ----------
        x : torch.Tensor
            Description.

        Returns
        -------
        torch.Tensor
            Description.

        """
        x = self.backbone(x)
        return self.head(x)


strategy = TorchTensorMulticlassClassificationStrategy(model_cls=MulticlassClassifier)
strategy.load(ARTIFACTS_DIRECTORY)
