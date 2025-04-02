import joblib
from sklearn.preprocessing import StandardScaler


class ScalerWrapper:
    def __init__(self):
        self.scaler = StandardScaler()

    def save_scaler(self, file_path):
        joblib.dump(self.scaler, file_path)

    def load_scaler(self, file_path):
        self.scaler = joblib.load(file_path)
