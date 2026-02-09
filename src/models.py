from pygam import LinearGAM, s
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
import numpy as np

class MeanPredictor:
    def fit(self, X, y):
        self.mean_ = y.mean()
    def predict(self, X):
        return np.full(shape=(X.shape[0],), fill_value=self.mean_)

# GAM
def train_gam(X_train, y_train):
    gam = LinearGAM(s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7))
    gam.fit(X_train, y_train)
    return gam

# Random Forest
def train_rf(X_train, y_train, n_estimators=200):
    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)
    return rf

# kNN
def train_knn(X_train, y_train, n_neighbors=5):
    knn = KNeighborsRegressor(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    return knn

# Generic predict
def predict(model, X):
    return model.predict(X)
