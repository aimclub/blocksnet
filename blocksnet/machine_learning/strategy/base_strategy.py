import os
import json
from abc import ABC, abstractmethod

META_FILENAME = "meta.json"

MODEL_CLS_KEY = "model_cls"
MODEL_PARAMS_KEY = "model_params"


class BaseStrategy(ABC):
    def __init__(self, model_cls, model_params: dict = {}):
        self.model = None
        self.model_cls = model_cls
        self.model_params = model_params

    @abstractmethod
    def train(self, *args, **kwargs):
        """Must call `super().train()` if overridden."""
        self.model = self.model_cls(**self.model_params)

    @abstractmethod
    def predict(self, *args, **kwargs):
        """Must call `super().predict()` if overridden."""
        if self.model is None:
            raise ValueError("Model is not initialized")

    @abstractmethod
    def validate(self, *args, **kwargs):
        """Must call `super().validate()` if overridden."""
        if self.model is None:
            raise ValueError("Model is not initialized")

    @abstractmethod
    def _save_model(self, path: str):
        pass

    @abstractmethod
    def _load_model(self, path: str):
        pass

    def update_model_params(self, **kwargs):
        self.model_params.update(kwargs)

    def _save_meta(self, path: str):
        meta = {MODEL_CLS_KEY: self.model_cls.__name__, MODEL_PARAMS_KEY: self.model_params}
        with open(os.path.join(path, META_FILENAME), "w") as f:
            json.dump(meta, f)

    def _load_meta(self, path: str):
        meta_path = os.path.join(path, META_FILENAME)
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Meta file not found at {meta_path}")
        with open(meta_path, "r") as f:
            meta = json.load(f)
        if meta[MODEL_CLS_KEY] != self.model_cls.__name__:
            raise ValueError(f"Model class mismatch: expected {self.model_cls.__name__}, got {meta[MODEL_CLS_KEY]}")
        self.model_params = meta.get(MODEL_PARAMS_KEY)

    def save(self, path: str):
        """Must call `super().save(path)` if overridden."""
        if self.model is None:
            raise ValueError("Model is not initialized")
        os.makedirs(path, exist_ok=True)
        self._save_meta(path)
        self._save_model(path)

    def load(self, path: str):
        """Must call `super().load(path)` if overridden."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path {path} does not exist")
        self._load_meta(path)
        self._load_model(path)
