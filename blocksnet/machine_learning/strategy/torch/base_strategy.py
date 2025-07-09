import os
from abc import abstractmethod
import joblib
import torch
import pandas as pd
from tqdm import tqdm
from ..base_strategy import BaseStrategy

TRAIN_LOSS_TEXT = "Train loss"
TEST_LOSS_TEXT = "Test loss"

MODEL_FILENAME = "model.pt"
SCALER_FILENAME = "scaler.joblib"

OPTIMIZER_CLS = torch.optim.Adam
OPTIMIZER_PARAMS = {"lr": 3e-4, "weight_decay": 1e-5}
CRITERION_CLS = torch.nn.MSELoss
CRITERION_PARAMS = {}


class TorchBaseStrategy(BaseStrategy):
    def __init__(self, model_cls: type[torch.nn.Module], model_params: dict | None = None, scaler=None, device=None):
        super().__init__(model_cls, model_params or {})
        self.scaler = scaler
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_optimizer(
        self, optimizer_cls: type[torch.optim.Optimizer] | None = None, optimizer_params: dict | None = None
    ):
        if optimizer_cls is None:
            optimizer_cls = OPTIMIZER_CLS
        if optimizer_params is None:
            optimizer_params = OPTIMIZER_PARAMS
        return optimizer_cls(self.model.parameters(), **optimizer_params)

    def _build_criterion(
        self, criterion_cls: type[torch.nn.Module] | None = None, criterion_params: dict | None = None
    ):
        if criterion_cls is None:
            criterion_cls = CRITERION_CLS
        if criterion_params is None:
            criterion_params = CRITERION_PARAMS
        return criterion_cls(**criterion_params)

    def _epoch(
        self, train: dict[str, torch.Tensor], test: dict[str, torch.Tensor], optimizer, criterion
    ) -> tuple[float, float]:
        model = self.model
        model.train()
        optimizer.zero_grad()
        train_out = model(train["x"])
        train_loss = criterion(train_out, train["y"])

    def _train_epoch(self, train: dict[str, torch.Tensor], optimizer, criterion) -> float:
        model = self.model
        model.train()
        optimizer.zero_grad()
        train_out = model(train["x"])
        train_loss = criterion(train_out, train["y"])
        train_loss.backward()
        optimizer.step()
        return train_loss.item()

    def _test_epoch(self, test: dict[str, torch.Tensor], criterion) -> float:
        test_out = self._predict(test["x"])
        test_loss = criterion(test_out, test["y"])
        return test_loss.item()

    def _train(
        self,
        train: dict[str, torch.Tensor],
        test: dict[str, torch.Tensor],
        epochs: int,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
    ) -> tuple[list[float], list[float]]:
        model = self.model
        train_losses = []
        test_losses = []
        pbar = tqdm(range(epochs), desc=f"{TRAIN_LOSS_TEXT}: ...... | {TEST_LOSS_TEXT}: ......")

        for _ in pbar:
            train_loss = self._train_epoch(train, optimizer, criterion)
            test_loss = self._test_epoch(test, criterion)

            train_losses.append(train_loss)
            test_losses.append(test_loss)

            pbar.set_description(f"{TRAIN_LOSS_TEXT}: {train_loss:.5f} | {TEST_LOSS_TEXT}: {test_loss:.5f}")

        return train_losses, test_losses

    def train(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        y_columns: list[str] | None = None,
        epochs: int = 100,
        optimizer_cls: type[torch.optim.Optimizer] | None = None,
        optimizer_params: dict | None = None,
        criterion_cls: type[torch.nn.Module] | None = None,
        criterion_params: dict | None = None,
    ) -> tuple[list[float], list[float]]:
        super().train()
        optimizer = self._build_optimizer(optimizer_cls, optimizer_params)
        criterion = self._build_criterion(criterion_cls, criterion_params)
        train = self.preprocess(train_df, y_columns)
        test = self.preprocess(test_df, y_columns)
        return self._train(train, test, epochs, optimizer, criterion)

    def _predict(self, x: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            pred = self.model(x)
        return pred

    def predict(self, df: pd.DataFrame, y_columns: list[str] | None = None) -> pd.DataFrame:
        super().predict()
        data = self.preprocess(df, y_columns)
        pred = self._predict(data["x"]).cpu()
        df = self.postprocess(pred, df.index.tolist(), df.columns.tolist() if y_columns is None else y_columns)
        return df

    def validate(
        self,
        df: pd.DataFrame,
        y_columns: list[str] | None = None,
        criterion_cls: type[torch.nn.Module] | None = None,
        criterion_params: dict | None = None,
    ) -> tuple[float, pd.DataFrame]:
        super().validate()
        data = self.preprocess(df, y_columns)
        pred = self._predict(data["x"]).cpu()
        criterion = self._build_criterion(criterion_cls, criterion_params)
        loss = criterion(pred, data["y"]).item()
        df = self.postprocess(pred, df.index.tolist(), df.columns.tolist() if y_columns is None else y_columns)
        return loss, df

    def preprocess(self, df: pd.DataFrame, y_columns: list[str] | None = None) -> dict[str, torch.Tensor]:
        if self.scaler is not None:
            df = self.scaler.transform(df)
        if y_columns is not None:
            x = df.drop(columns=y_columns).values
            y = df[y_columns].values
            return {
                "x": torch.tensor(x, dtype=torch.float).to(self.device),
                "y": torch.tensor(y, dtype=torch.float).to(self.device),
            }
        x = df.values
        return {"x": torch.tensor(x, dtype=torch.float).to(self.device)}

    def postprocess(self, tensor: torch.Tensor, index: list[int], columns: list[str]) -> pd.DataFrame:
        array = tensor.detach().cpu().numpy()
        df = pd.DataFrame(array, index=index, columns=columns)
        if self.scaler is not None:
            df = self.scaler.inverse_transform(df)
        return df

    def _save_model(self, path: str):
        state_dict = self.model.state_dict()
        torch.save(state_dict, os.path.join(path, MODEL_FILENAME))

    def _load_model(self, path: str):
        model_path = os.path.join(path, MODEL_FILENAME)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        state_dict = torch.load(model_path)
        self.model = self.model_cls(**self.model_params)
        self.model.load_state_dict(state_dict)

    def _save_scaler(self, path: str):
        if self.scaler is not None:
            joblib.dump(self.scaler, os.path.join(path, SCALER_FILENAME))

    def _load_scaler(self, path: str):
        self.scaler = None
        scaler_path = os.path.join(path, SCALER_FILENAME)
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)

    def save(self, path: str):
        super().save(path)
        self._save_scaler(path)

    def load(self, path: str):
        super().load(path)
        self._load_scaler(path)
