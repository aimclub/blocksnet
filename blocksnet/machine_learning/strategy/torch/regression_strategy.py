import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from blocksnet.config import log_config
from .base_strategy import TorchBaseStrategy

TRAIN_LOSS_TEXT = "Train loss"
TEST_LOSS_TEXT = "Test loss"


class TorchRegressionStrategy(TorchBaseStrategy):
    def _build_optimizer(
        self, optimizer_cls: type[torch.optim.Optimizer] | None = None, optimizer_params: dict | None = None
    ):
        optimizer_cls = optimizer_cls or torch.optim.Adam
        optimizer_params = optimizer_params or {"lr": 3e-4, "weight_decay": 1e-5}
        return super()._build_optimizer(optimizer_cls, optimizer_params)

    def _build_criterion(
        self, criterion_cls: type[torch.nn.Module] | None = None, criterion_params: dict | None = None
    ):
        criterion_cls = criterion_cls or torch.nn.MSELoss
        criterion_params = criterion_params or {}
        return super()._build_criterion(criterion_cls, criterion_params)

    def _epoch_train(self, x_train, y_train, optimizer, criterion):
        model = self.model
        model.train()
        optimizer.zero_grad()
        y_pred = model(x_train)
        loss = criterion(y_train, y_pred)
        loss.backward()
        optimizer.step()
        return loss.item()

    def _epoch_test(self, x_test, y_test, criterion):
        model = self.model
        model.eval()
        with torch.no_grad():
            y_pred = model(x_test)
        loss = criterion(y_test, y_pred)
        return loss.item()

    def train(
        self,
        x_train: np.ndarray,
        x_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        optimizer_cls: type[torch.optim.Optimizer] | None = None,
        optimizer_params: dict | None = None,
        criterion_cls: type[torch.nn.Module] | None = None,
        criterion_params: dict | None = None,
    ) -> tuple[list[float], list[float]]:

        model = self._build_model(x_train.shape[1], y_train.shape[1])
        model.to(self.device)

        optimizer = self._build_optimizer(optimizer_cls, optimizer_params)
        criterion = self._build_criterion(criterion_cls, criterion_params)

        x_train_tensor, x_test_tensor = self._preprocess("x", True, x_train, x_test)
        y_train_tensor, y_test_tensor = self._preprocess("y", True, y_train, y_test)

        dataset = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        train_losses = []
        test_losses = []
        pbar = tqdm(
            range(epochs), disable=log_config.disable_tqdm, desc=f"{TRAIN_LOSS_TEXT}: ...... | {TEST_LOSS_TEXT}: ......"
        )
        for _ in pbar:
            batch_losses = []
            for x_batch, y_batch in dataloader:
                batch_loss = self._epoch_train(x_batch, y_batch, optimizer, criterion)
                batch_losses.append(batch_loss)

            train_loss = np.mean(batch_losses)
            test_loss = self._epoch_test(x_test_tensor, y_test_tensor, criterion)

            train_losses.append(train_loss)
            test_losses.append(test_loss)

            pbar.set_description(f"{TRAIN_LOSS_TEXT}: {train_loss:.5f} | {TEST_LOSS_TEXT}: {test_loss:.5f}")

        return train_losses, test_losses

    def predict(self, x: np.ndarray) -> np.ndarray:
        model = self.model.to(self.device)
        model.eval()
        with torch.no_grad():
            (x_tensor,) = self._preprocess("x", False, x)
            y_pred_tensor = model(x_tensor)
            (y_pred,) = self._postprocess("y", y_pred_tensor)
        return y_pred
