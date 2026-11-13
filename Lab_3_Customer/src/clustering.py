
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering


# 1. K-Means Clustering
def run_kmeans(X_scaled, n_clusters=5):
    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_scaled)
    return model, labels


# 2. DBSCAN Clustering
def run_dbscan(X_scaled, eps=0.5, min_samples=5):
    model = DBSCAN(
        eps=eps,
        min_samples=min_samples
    )

    labels = model.fit_predict(X_scaled)
    return model, labels


# 3. Hierarchical Clustering
def run_hierarchical(X_scaled, n_clusters=5):
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage="ward"
    )

    labels = model.fit_predict(X_scaled)
    return model, labels
