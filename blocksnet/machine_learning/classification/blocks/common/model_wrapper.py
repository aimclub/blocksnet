from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score

from .....config import log_config


TRAIN_LOSS_TEXT = "Train loss"
TEST_LOSS_TEXT = "Test loss"


class ModelWrapper:
    def __init__(self, path: str, *args, **kwargs):
        self.model = CatBoostClassifier(*args, **kwargs)
        self.load_model(path)

    def load_model(self, file_path: str, *args, **kwargs):
        self.model.load_model(file_path, *args, **kwargs)

    def save_model(self, file_path: str, *args, **kwargs):
        self.model.save_model(file_path, *args, **kwargs)

    def _train_model(self, x_train, y_train, **kwargs):
        model = CatBoostClassifier(loss_function="MultiClass", **kwargs)
        model.fit(x_train, y_train)
        self.model = model

    def _evaluate_model(self, x):
        return self.model.predict_proba(x)

    def _test_model(self, x_test, y_test):
        y_pred = self._evaluate_model(x_test)
        return accuracy_score(y_test, y_pred)
