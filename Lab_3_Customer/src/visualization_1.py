import matplotlib.pyplot as plt
import numpy as np


# 1. Elbow Method
def plot_elbow(inertia_values, save_path=None):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(inertia_values) + 1), inertia_values, marker="o")
    plt.title("Elbow Method")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia")
    plt.xticks(range(1, len(inertia_values) + 1))
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


# 2. Clustering Visualization
def plot_clusters(X_scaled, labels, title, save_path=None):
    values = X_scaled.to_numpy() if hasattr(X_scaled, "to_numpy") else np.asarray(X_scaled)
    plt.figure(figsize=(8, 5))
    plt.scatter(values[:, 0], values[:, 1], c=labels, cmap="viridis", s=50)
    plt.title(title)
    plt.xlabel("Annual Income (Scaled)")
    plt.ylabel("Spending Score (Scaled)")
    plt.colorbar(label="Cluster")
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


# 3. K-distance Graph for DBSCAN - tự tính khoảng cách, không dùng NearestNeighbors.
def plot_k_distance(X_scaled, n_neighbors=5, save_path=None):
    values = X_scaled.to_numpy() if hasattr(X_scaled, "to_numpy") else np.asarray(X_scaled)
    if n_neighbors < 1 or n_neighbors > len(values):
        raise ValueError("n_neighbors không hợp lệ.")
    distances = np.sqrt(np.sum((values[:, None, :] - values[None, :, :]) ** 2, axis=2))
    sorted_distances = np.sort(distances, axis=1)
    kth_distances = np.sort(sorted_distances[:, n_neighbors - 1])

    plt.figure(figsize=(8, 5))
    plt.plot(kth_distances)
    plt.title("K-distance Graph for DBSCAN")
    plt.xlabel("Data Points (sorted)")
    plt.ylabel(f"Distance to {n_neighbors}th Nearest Neighbor")
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
