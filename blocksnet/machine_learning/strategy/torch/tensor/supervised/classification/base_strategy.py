import torch
import numpy as np
from abc import ABC, abstractmethod
from loguru import logger
from blocksnet.machine_learning.strategy.classification_base import ClassificationBase
from ..base_strategy import TorchTensorSupervisedStrategy

CRITERIONS = {
    torch.nn.CrossEntropyLoss: {
        "pred": lambda logits: torch.argmax(logits, dim=1),
        "prob": lambda logits: torch.softmax(logits, dim=1),
    },
    torch.nn.NLLLoss: {
        "pred": lambda logits: torch.argmax(logits, dim=1),
        "prob": lambda logits: torch.exp(logits),
    },
    torch.nn.BCEWithLogitsLoss: {
        "pred": lambda logits: (torch.sigmoid(logits) > 0.5).int(),
        "prob": lambda logits: torch.sigmoid(logits),
    },
}


class TorchTensorBaseClassificationStrategy(ClassificationBase, TorchTensorSupervisedStrategy, ABC):

    _default_criterion_cls: type[torch.nn.Module]

    def _x_to_tensor(self, x: np.ndarray) -> torch.Tensor:
        return torch.tensor(x, dtype=torch.float32, device=self.device)

    def _build_criterion(
        self,
        criterion_cls: type[torch.nn.Module] | None,
        criterion_params: dict | None,
    ) -> torch.nn.Module:
        if criterion_cls is None:
            criterion_cls = self._default_criterion_cls
            logger.warning(f"One should provide criterion_cls. Using {criterion_cls.__name__} by default.")
        if not criterion_cls in CRITERIONS:
            logger.warning(
                f"Provided criterion_cls {criterion_cls.__name__} is not in known CRITERIONS. "
                "predict() and predict_proba() may behave unexpectedly."
            )
        criterion_params = criterion_params or {}
        return criterion_cls(**criterion_params)

    def predict(self, x: np.ndarray, criterion_cls: type[torch.nn.Module] | None = None):
        logits = super().predict(x)
        logits = torch.from_numpy(logits)
        if criterion_cls is None:
            criterion_cls = self._default_criterion_cls
            logger.warning(f"One should provide criterion_cls. Using {criterion_cls.__name__} by default.")
        if criterion_cls in CRITERIONS:
            func = CRITERIONS[criterion_cls]["pred"]
            preds = func(logits)
        else:
            logger.warning(f"Not implemented for provided criterion_cls, using argmax as fallback.")
            preds = torch.argmax(logits, dim=1)
        return preds.numpy()

    def predict_proba(self, x: np.ndarray, criterion_cls: type[torch.nn.Module] | None = None) -> np.ndarray:
        logits = super().predict(x)
        logits = torch.from_numpy(logits)
        if criterion_cls is None:
            criterion_cls = self._default_criterion_cls
            logger.warning(f"One should provide criterion_cls. Using {criterion_cls.__name__} by default.")
        if criterion_cls in CRITERIONS:
            func = CRITERIONS[criterion_cls]["prob"]
            probs = func(logits)
        else:
            logger.warning(f"Not implemented for provided criterion_cls, using softmax as fallback.")
            probs = torch.softmax(logits, dim=1)
        return probs.numpy()
