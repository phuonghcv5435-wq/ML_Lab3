
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler


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


# 3. Scale features
def scale_features(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_scaled = pd.DataFrame(
        X_scaled,
        columns=X.columns,
        index=X.index
    )

    return X_scaled, scaler
