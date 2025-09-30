import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from abc import ABC, abstractmethod
from ..base_strategy import TorchTensorBaseStrategy


class TorchTensorSupervisedStrategy(TorchTensorBaseStrategy, ABC):
    """TorchTensorSupervisedStrategy class.

    """
    def _epoch_train(
        self, x_train: torch.Tensor, y_train: torch.Tensor, optimizer: torch.optim.Optimizer, criterion: torch.nn.Module
    ) -> float:
        """Epoch train.

        Parameters
        ----------
        x_train : torch.Tensor
            Description.
        y_train : torch.Tensor
            Description.
        optimizer : torch.optim.Optimizer
            Description.
        criterion : torch.nn.Module
            Description.

        Returns
        -------
        float
            Description.

        """
        model = self.model
        model.train()
        optimizer.zero_grad()
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
        return loss.item()

    def _epoch_test(self, x_test: torch.Tensor, y_test: torch.Tensor, criterion: torch.nn.Module) -> float:
        """Epoch test.

        Parameters
        ----------
        x_test : torch.Tensor
            Description.
        y_test : torch.Tensor
            Description.
        criterion : torch.nn.Module
            Description.

        Returns
        -------
        float
            Description.

        """
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

        """Epoch.

        Parameters
        ----------
        data_loader : DataLoader
            Description.
        data : dict[str, torch.Tensor]
            Description.
        optimizer : torch.optim.Optimizer
            Description.
        criterion : torch.nn.Module
            Description.
        **kwargs : dict
            Description.

        Returns
        -------
        tuple[float, float]
            Description.

        """
        batch_losses = []
        for x_batch, y_batch in data_loader:
            batch_loss = self._epoch_train(x_batch, y_batch, optimizer, criterion)
            batch_losses.append(batch_loss)

        train_loss = float(np.mean(batch_losses))
        test_loss = self._epoch_test(data["x_test"], data["y_test"], criterion)

        return train_loss, test_loss

    def _predict(self, data: dict[str, torch.Tensor]) -> torch.Tensor:
        """Predict.

        Parameters
        ----------
        data : dict[str, torch.Tensor]
            Description.

        Returns
        -------
        torch.Tensor
            Description.

        """
        return self.model(data["x"])

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict.

        Parameters
        ----------
        x : np.ndarray
            Description.

        Returns
        -------
        np.ndarray
            Description.

        """
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
        """Train.

        Parameters
        ----------
        x_train : np.ndarray
            Description.
        x_test : np.ndarray
            Description.
        y_train : np.ndarray
            Description.
        y_test : np.ndarray
            Description.
        epochs : int, default: 100
            Description.
        optimizer_cls : type[torch.optim.Optimizer] | None, default: None
            Description.
        optimizer_params : dict | None, default: None
            Description.
        criterion_cls : type[torch.nn.Module] | None, default: None
            Description.
        criterion_params : dict | None, default: None
            Description.
        data_loader_params : dict | None, default: None
            Description.

        Returns
        -------
        tuple[list[float], list[float]]
            Description.

        """
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
        """X to tensor.

        Parameters
        ----------
        x : np.ndarray
            Description.

        Returns
        -------
        torch.Tensor
            Description.

        """
        pass

    @abstractmethod
    def _y_to_tensor(self, y: np.ndarray) -> torch.Tensor:
        """Y to tensor.

        Parameters
        ----------
        y : np.ndarray
            Description.

        Returns
        -------
        torch.Tensor
            Description.

        """
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

        """Preprocess.

        Parameters
        ----------
        x : np.ndarray | None, default: None
            Description.
        x_train : np.ndarray | None, default: None
            Description.
        x_test : np.ndarray | None, default: None
            Description.
        y : np.ndarray | None, default: None
            Description.
        y_train : np.ndarray | None, default: None
            Description.
        y_test : np.ndarray | None, default: None
            Description.

        Returns
        -------
        dict[str, torch.Tensor]
            Description.

        """
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
        """Postprocess.

        Parameters
        ----------
        y : torch.Tensor
            Description.

        Returns
        -------
        np.ndarray
            Description.

        """
        a = y.cpu().numpy()
        (a,) = self._inverse_scale("y", a)
        return a
