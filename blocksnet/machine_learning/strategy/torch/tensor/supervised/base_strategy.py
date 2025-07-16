import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from abc import ABC, abstractmethod
from ..base_strategy import TorchTensorBaseStrategy


class TorchTensorSupervisedStrategy(TorchTensorBaseStrategy, ABC):
    def _epoch_train(
        self, x_train: torch.Tensor, y_train: torch.Tensor, optimizer: torch.optim.Optimizer, criterion: torch.nn.Module
    ) -> float:
        model = self.model
        model.train()
        optimizer.zero_grad()
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
        return loss.item()

    def _epoch_test(self, x_test: torch.Tensor, y_test: torch.Tensor, criterion: torch.nn.Module) -> float:
        model = self.model
        model.eval()
        with torch.no_grad():
            y_pred = model(x_test)
        loss = criterion(y_pred, y_test)
        return loss.item()

    def _epoch(
        self,
        data_loader: DataLoader,
        data: dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        **kwargs,
    ) -> tuple[float, float]:

        batch_losses = []
        for x_batch, y_batch in data_loader:
            batch_loss = self._epoch_train(x_batch, y_batch, optimizer, criterion)
            batch_losses.append(batch_loss)

        train_loss = float(np.mean(batch_losses))
        test_loss = self._epoch_test(data["x_test"], data["y_test"], criterion)

        return train_loss, test_loss

    def _predict(self, data: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.model(data["x"])

    def predict(self, x: np.ndarray) -> np.ndarray:
        return super().predict(x=x)

    def train(
        self,
        x_train: np.ndarray,
        x_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        epochs: int = 100,
        optimizer_cls: type[torch.optim.Optimizer] | None = None,
        optimizer_params: dict | None = None,
        criterion_cls: type[torch.nn.Module] | None = None,
        criterion_params: dict | None = None,
        data_loader_params: dict | None = None,
    ) -> tuple[list[float], list[float]]:
        return super().train(
            x_train=x_train,
            x_test=x_test,
            y_train=y_train,
            y_test=y_test,
            epochs=epochs,
            optimizer_cls=optimizer_cls,
            optimizer_params=optimizer_params,
            criterion_cls=criterion_cls,
            criterion_params=criterion_params,
            data_loader_params=data_loader_params,
        )

    @abstractmethod
    def _x_to_tensor(self, x: np.ndarray) -> torch.Tensor:
        pass

    @abstractmethod
    def _y_to_tensor(self, y: np.ndarray) -> torch.Tensor:
        pass

    def _preprocess(
        self,
        x: np.ndarray | None = None,
        x_train: np.ndarray | None = None,
        x_test: np.ndarray | None = None,
        y: np.ndarray | None = None,
        y_train: np.ndarray | None = None,
        y_test: np.ndarray | None = None,
    ) -> dict[str, torch.Tensor]:

        result: dict[str, torch.Tensor] = {}

        if x_train is not None:
            if x_test is None:
                raise ValueError("x_test must not be None while x_train is set")
            x_train, x_test = self._scale("x", True, x_train, x_test)
            result.update({"x_train": self._x_to_tensor(x_train), "x_test": self._x_to_tensor(x_test)})

        if x is not None:
            (x,) = self._scale("x", False, x)
            result.update({"x": self._x_to_tensor(x)})

        if y_train is not None:
            if y_test is None:
                raise ValueError("y_test must not be None while y_train is set")
            y_train, y_test = self._scale("y", True, y_train, y_test)
            result.update({"y_train": self._y_to_tensor(y_train), "y_test": self._y_to_tensor(y_test)})

        if y is not None:
            (y,) = self._scale("y", False, y)
            result.update({"y": self._x_to_tensor(y)})

        return result

    def _postprocess(self, y: torch.Tensor) -> np.ndarray:
        a = y.cpu().numpy()
        (a,) = self._inverse_scale("y", a)
        return a
