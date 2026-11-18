import numpy as np

# 1. K-Means Clustering
class KMeansManual:
    def __init__(self, n_clusters=5, max_iter=100, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)

        np.random.seed(self.random_state)

        # Chọn ngẫu nhiên các tâm cụm ban đầu
        indices = np.random.choice(
            len(X), self.n_clusters, replace=False
        )
        centers = X[indices].copy()

        for _ in range(self.max_iter):
            distances = np.linalg.norm(
                X[:, np.newaxis] - centers, axis=2
            )

            labels = np.argmin(distances, axis=1)

            new_centers = centers.copy()

            for i in range(self.n_clusters):
                points = X[labels == i]

                if len(points) > 0:
                    new_centers[i] = points.mean(axis=0)

            if np.allclose(centers, new_centers):
                centers = new_centers
                break

            centers = new_centers

    
        distances = np.linalg.norm(
            X[:, np.newaxis] - centers, axis=2
        )
        labels = np.argmin(distances, axis=1)

        self.cluster_centers_ = centers
        self.labels_ = labels

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)

        distances = np.linalg.norm(
            X[:, np.newaxis] - self.cluster_centers_,
            axis=2
        )

        return np.argmin(distances, axis=1)


# 2. DBSCAN Clustering
class DBSCANManual:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)

        n_samples = len(X)
        labels = np.full(n_samples, -1)
        visited = np.zeros(n_samples, dtype=bool)

        cluster_id = 0

        # Tính khoảng cách giữa các điểm
        distances = np.linalg.norm(
            X[:, np.newaxis] - X[np.newaxis, :],
            axis=2
        )

        for i in range(n_samples):
            if visited[i]:
                continue

            visited[i] = True

            neighbors = np.where(
                distances[i] <= self.eps
            )[0]

            if len(neighbors) < self.min_samples:
                continue

            labels[i] = cluster_id

            seeds = list(neighbors)
            j = 0

            while j < len(seeds):
                point = seeds[j]

                if not visited[point]:
                    visited[point] = True

                    point_neighbors = np.where(
                        distances[point] <= self.eps
                    )[0]

                    if len(point_neighbors) >= self.min_samples:
                        for neighbor in point_neighbors:
                            if neighbor not in seeds:
                                seeds.append(neighbor)

                if labels[point] == -1:
                    labels[point] = cluster_id

                j += 1

            cluster_id += 1

        self.labels_ = labels
        return self


# 3. Hierarchical Clustering
class HierarchicalManual:
    def __init__(self, n_clusters=5):
        self.n_clusters = n_clusters
        self.labels_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)

        n_samples = len(X)

        if self.n_clusters < 1 or self.n_clusters > n_samples:
            raise ValueError("So cum khong hop le")

        clusters = [[i] for i in range(n_samples)]

        while len(clusters) > self.n_clusters:
            min_distance = float("inf")
            merge_a = 0
            merge_b = 1

            # Tìm hai cụm gần nhau nhất
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    distances = []

                    for a in clusters[i]:
                        for b in clusters[j]:
                            distance = np.linalg.norm(X[a] - X[b])
                            distances.append(distance)

                    average_distance = np.mean(distances)

                    if average_distance < min_distance:
                        min_distance = average_distance
                        merge_a = i
                        merge_b = j

            # Gộp hai cụm lại
            clusters[merge_a].extend(clusters[merge_b])
            clusters.pop(merge_b)

        labels = np.zeros(n_samples, dtype=int)

        for cluster_id, cluster in enumerate(clusters):
            for index in cluster:
                labels[index] = cluster_id

        self.labels_ = labels
        return self


# Hàm chạy K-Means
def run_kmeans(X_scaled, n_clusters=5):
    model = KMeansManual(n_clusters=n_clusters)
    model.fit(X_scaled)
    return model, model.labels_


# Hàm chạy DBSCAN
def run_dbscan(X_scaled, eps=0.5, min_samples=5):
    model = DBSCANManual(
        eps=eps,
        min_samples=min_samples
    )
    model.fit(X_scaled)
    return model, model.labels_


# Hàm chạy Hierarchical
def run_hierarchical(X_scaled, n_clusters=5):
    model = HierarchicalManual(n_clusters=n_clusters)
    model.fit(X_scaled)
    return model, model.labels_
