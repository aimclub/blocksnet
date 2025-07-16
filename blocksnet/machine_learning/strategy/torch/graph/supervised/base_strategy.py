import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from abc import ABC, abstractmethod
from ..base_strategy import TorchGraphBaseStrategy


class TorchGraphSupervisedStrategy(TorchGraphBaseStrategy, ABC):
    def _epoch_train(self, batch: Data, optimizer: torch.optim.Optimizer, criterion: torch.nn.Module) -> float:
        model = self.model
        model.train()
        optimizer.zero_grad()
        y_pred = model(x=batch.x, edge_index=batch.edge_index, edge_attr=batch.edge_attr)
        loss = criterion(y_pred[batch.train_mask], batch.y[batch.train_mask])
        loss.backward()
        optimizer.step()
        return loss.item()

    def _epoch_test(self, batch: Data, criterion: torch.nn.Module) -> float:
        model = self.model
        model.eval()
        with torch.no_grad():
            y_pred = model(x=batch.x, edge_index=batch.edge_index, edge_attr=batch.edge_attr)
        loss = criterion(y_pred[batch.test_mask], batch.y[batch.test_mask])
        return loss.item()

    def _epoch(
        self,
        data_loader: DataLoader,
        data: dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        **kwargs,
    ) -> tuple[float, float]:

        batch_train_losses = []
        batch_test_losses = []

        for batch in data_loader:
            batch_train_loss = self._epoch_train(batch, optimizer, criterion)
            batch_train_losses.append(batch_train_loss)
            # test
            batch_test_loss = self._epoch_test(batch, criterion)
            batch_test_losses.append(batch_test_loss)

        train_loss = float(np.mean(batch_train_losses))
        test_loss = float(np.mean(batch_test_losses))

        return train_loss, test_loss

    def _predict(self, data: dict[str, torch.Tensor]) -> torch.Tensor:
        x = data["x"]
        edge_index = data["edge_index"]
        edge_attr = data.get("edge_attr")
        return self.model(x, edge_index, edge_attr)

    def predict(self, x: np.ndarray, edge_index: np.ndarray, edge_attr: np.ndarray | None) -> np.ndarray:
        return super().predict(x=x, edge_index=edge_index, edge_attr=edge_attr)

    def train(
        self,
        x: np.ndarray,
        y: np.ndarray,
        edge_index: np.ndarray,
        train_mask: np.ndarray,
        test_mask: np.ndarray,
        edge_attr: np.ndarray | None = None,
        epochs: int = 100,
        optimizer_cls: type[torch.optim.Optimizer] | None = None,
        optimizer_params: dict | None = None,
        criterion_cls: type[torch.nn.Module] | None = None,
        criterion_params: dict | None = None,
        data_loader_params: dict | None = None,
    ) -> tuple[list[float], list[float]]:
        return super().train(
            x=x,
            y=y,
            edge_index=edge_index,
            train_mask=train_mask,
            test_mask=test_mask,
            edge_attr=edge_attr,
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

    def _edge_index_to_tensor(self, edge_index: np.ndarray) -> torch.Tensor:
        return torch.tensor(edge_index, dtype=torch.long, device=self.device)

    @abstractmethod
    def _edge_attr_to_tensor(self, edge_attr: np.ndarray) -> torch.Tensor:
        pass

    def _preprocess(
        self,
        x: np.ndarray,
        edge_index: np.ndarray,
        y: np.ndarray | None = None,
        edge_attr: np.ndarray | None = None,
        train_mask: np.ndarray | None = None,
        test_mask: np.ndarray | None = None,
    ) -> dict[str, torch.Tensor]:

        result = {}
        is_fit_scaler = train_mask is not None

        (x,) = self._scale("x", is_fit_scaler, x)
        result["x"] = self._x_to_tensor(x)
        result["edge_index"] = self._edge_index_to_tensor(edge_index)

        if y is not None:
            (y,) = self._scale("y", is_fit_scaler, y)
            result["y"] = self._y_to_tensor(y)

        if edge_attr is not None:
            (edge_attr,) = self._scale("edge_attr", is_fit_scaler, edge_attr)
            result["edge_attr"] = self._edge_attr_to_tensor(edge_attr)

        if train_mask is not None:
            result["train_mask"] = torch.tensor(train_mask, dtype=torch.bool, device=self.device)
            result["test_mask"] = torch.tensor(test_mask, dtype=torch.bool, device=self.device)

        return result

    def _postprocess(self, y: torch.Tensor) -> np.ndarray:
        a = y.cpu().numpy()
        (a,) = self._inverse_scale("y", a)
        return a
