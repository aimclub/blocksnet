import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler

class ScalerWrapper:
    def __init__(self):
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()

    def fit(self, x, y):
        self.scaler_x.fit(x)
        self.scaler_y.fit(y)

    def transform(self, x, y):
        X_scaled = self.scaler_x.transform(x)
        y_scaled = self.scaler_y.transform(y)
        return X_scaled, y_scaled

    def inverse_transform_y(self, y_scaled):
        return self.scaler_y.inverse_transform(y_scaled)
    
    def inverse_transform_x(self, x_scaled):
        return self.scaler_x.inverse_transform(x_scaled)

    def save_scalers(self, path_x, path_y):
        joblib.dump(self.scaler_x, path_x)
        joblib.dump(self.scaler_y, path_y)

    def load_scalers(self, path_x, path_y):
        self.scaler_x = joblib.load(path_x)
        self.scaler_y = joblib.load(path_y)
