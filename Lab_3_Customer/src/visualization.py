import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import numpy as np

# 1. Elbow Method
def plot_elbow(inertia_values, save_path=None):
    plt.figure(figsize=(8, 5))

    plt.plot(
        range(1, len(inertia_values) + 1),
        inertia_values,
        marker="o"
    )

    plt.title("Elbow Method")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia")
    plt.xticks(range(1, len(inertia_values) + 1))
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


# 2. Clustering Visualization
def plot_clusters(X_scaled, labels, title, save_path=None):
    plt.figure(figsize=(8, 5))

    plt.scatter(
        X_scaled.iloc[:, 0],
        X_scaled.iloc[:, 1],
        c=labels,
        cmap="viridis",
        s=50
    )

    plt.title(title)
    plt.xlabel("Annual Income (Scaled)")
    plt.ylabel("Spending Score (Scaled)")
    plt.colorbar(label="Cluster")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


# 3. K-distance Graph for DBSCAN
def plot_k_distance(X_scaled, n_neighbors=5, save_path=None):
    neighbors = NearestNeighbors(n_neighbors=n_neighbors)

    neighbors_fit = neighbors.fit(X_scaled)

    distances, indices = neighbors_fit.kneighbors(X_scaled)

    distances = np.sort(distances[:, n_neighbors - 1])

    plt.figure(figsize=(8, 5))
    plt.plot(distances)

    plt.title("K-distance Graph for DBSCAN")
    plt.xlabel("Data Points (sorted)")
    plt.ylabel(f"Distance to {n_neighbors}th Nearest Neighbor")

    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()