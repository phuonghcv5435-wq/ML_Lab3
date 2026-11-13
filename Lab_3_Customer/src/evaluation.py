
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# 1. Elbow Method
def calculate_inertia(X_scaled, max_k=10):
    inertia_values = []

    for k in range(1, max_k + 1):
        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        model.fit(X_scaled)
        inertia_values.append(model.inertia_)

    return inertia_values


# 2. Silhouette Score
def calculate_silhouette(X_scaled, labels):
    labels = list(labels)

    # Loại bỏ noise của DBSCAN
    mask = [label != -1 for label in labels]

    X_evaluate = X_scaled.loc[mask] if hasattr(X_scaled, "loc") else X_scaled[mask]
    labels_evaluate = [label for label in labels if label != -1]

    unique_clusters = set(labels_evaluate)

    if len(unique_clusters) < 2:
        return None

    return silhouette_score(X_evaluate, labels_evaluate)
