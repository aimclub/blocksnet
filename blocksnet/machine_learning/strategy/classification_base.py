from abc import ABC, abstractmethod


class ClassificationBase(ABC):
    @abstractmethod
    """ClassificationBase class.

    """
    def predict_proba(self, *args, **kwargs):
        """Predict proba.

        Parameters
        ----------
        *args : tuple
            Description.
        **kwargs : dict
            Description.

        """
        pass
