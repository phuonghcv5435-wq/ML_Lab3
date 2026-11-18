import pandas as pd
import numpy as np
from pathlib import Path


# 1. Load dataset
def load_data():
    project_dir = Path(__file__).resolve().parent.parent
    data_path = project_dir / "data" / "raw" / "Mall_Customers.csv"
    df = pd.read_csv(data_path)
    return df


# 2. Select features
def select_features(df):
    X = df[["Annual Income (k$)", "Spending Score (1-100)"]]
    return X


class StandardScalerManual:
    """Chuẩn hóa dữ liệu theo công thức (x - mean) / standard deviation."""
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        values = np.asarray(X, dtype=float)
        self.mean_ = values.mean(axis=0)
        self.scale_ = values.std(axis=0, ddof=0)
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, X):
        if self.mean_ is None:
            raise ValueError("Cần fit dữ liệu trước khi transform.")
        values = np.asarray(X, dtype=float)
        return (values - self.mean_) / self.scale_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        if self.mean_ is None:
            raise ValueError("Cần fit dữ liệu trước khi inverse_transform.")
        return np.asarray(X, dtype=float) * self.scale_ + self.mean_


# 3. Scale features
def scale_features(X):
    scaler = StandardScalerManual()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(
        X_scaled,
        columns=X.columns,
        index=X.index
    )
    return X_scaled, scaler
