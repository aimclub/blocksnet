import joblib
from sklearn.preprocessing import StandardScaler


class ScalerWrapper:
    """ScalerWrapper class.

    """
    def __init__(self, path: str):
        """Initialize the instance.

        Parameters
        ----------
        path : str
            Description.

        Returns
        -------
        None
            Description.

        """
        self.scaler = StandardScaler()
        self.load_scaler(path)

    def save_scaler(self, file_path):
        """Save scaler.

        Parameters
        ----------
        file_path : Any
            Description.

        """
        joblib.dump(self.scaler, file_path)

    def load_scaler(self, file_path):
        """Load scaler.

        Parameters
        ----------
        file_path : Any
            Description.

        """
        self.scaler = joblib.load(file_path)
