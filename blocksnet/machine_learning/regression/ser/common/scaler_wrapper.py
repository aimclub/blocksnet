import joblib
from sklearn.preprocessing import MinMaxScaler

class ScalerWrapper:
    def __init__(self):
        self.scaler_X = MinMaxScaler()
        self.scaler_Y = MinMaxScaler()

    def fit(self, X, Y):
        self.scaler_X.fit(X)
        self.scaler_Y.fit(Y)

    def transform(self, X, Y):
        X_scaled = self.scaler_X.transform(X)
        Y_scaled = self.scaler_Y.transform(Y)
        return X_scaled, Y_scaled

    def inverse_transform_Y(self, Y_scaled):
        return self.scaler_Y.inverse_transform(Y_scaled)

    def save_scalers(self, path_X, path_Y):
        joblib.dump(self.scaler_X, path_X)
        joblib.dump(self.scaler_Y, path_Y)

    def load_scalers(self, path_X, path_Y):
        self.scaler_X = joblib.load(path_X)
        self.scaler_Y = joblib.load(path_Y)
