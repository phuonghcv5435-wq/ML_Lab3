import numpy as np
from .clustering import run_kmeans


# 1. Elbow Method - tính tổng bình phương khoảng cách trong cụm.
def calculate_inertia(X_scaled, max_k=10):
    inertia_values = []
    for k in range(1, max_k + 1):
        model, _ = run_kmeans(X_scaled, n_clusters=k)
        inertia_values.append(model.inertia_)
    return inertia_values


# 2. Silhouette Score tự cài đặt, không dùng sklearn.metrics.
def calculate_silhouette(X_scaled, labels):
    if hasattr(X_scaled, "to_numpy"):
        X = X_scaled.to_numpy(dtype=float)
    else:
        X = np.asarray(X_scaled, dtype=float)
    labels = np.asarray(labels)

    # Không tính các điểm nhiễu (-1) của DBSCAN.
    mask = labels != -1
    X = X[mask]
    labels = labels[mask]
    unique_labels = np.unique(labels)

    if len(unique_labels) < 2 or len(unique_labels) >= len(labels):
        return None

    # Tính khoảng cách Euclid giữa từng cặp điểm.
    distances = np.sqrt(np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2))
    scores = []

    for i in range(len(X)):
        same_cluster = labels == labels[i]
        same_cluster[i] = False
        if not np.any(same_cluster):
            continue
        a = distances[i, same_cluster].mean()

        b = float("inf")
        for other_label in unique_labels:
            if other_label == labels[i]:
                continue
            other_cluster = labels == other_label
            b = min(b, distances[i, other_cluster].mean())

        denominator = max(a, b)
        scores.append(0.0 if denominator == 0 else (b - a) / denominator)

    if not scores:
        return None
    return float(np.mean(scores))
