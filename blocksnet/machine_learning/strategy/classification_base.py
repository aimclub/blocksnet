from abc import ABC, abstractmethod


class ClassificationBase(ABC):
    @abstractmethod
    def predict_proba(self, *args, **kwargs):
        pass
