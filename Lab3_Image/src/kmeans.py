import numpy as np


def kmeans(X, k, max_iter=100, tol=1e-4, seed=42):
    """K-Means tự cài đặt (Phiên bản tối ưu vector hóa).

    X: mảng (N, D) - mỗi hàng là một vector pixel (float32)
    k: số lượng cụm màu
    Trả về: centroids (k, D), labels (N,)
    """
    rng = np.random.default_rng(seed)

    # 1. Khởi tạo K centroid ngẫu nhiên từ tập dữ liệu X
    centroids = X[rng.choice(len(X), size=k, replace=False)].astype(np.float32)

    for _ in range(max_iter):
        # 2. Gán mỗi pixel vào centroid gần nhất dùng bình phương khoảng cách Euclid
        # (N, 1, D) - (1, k, D) -> (N, k, D) -> sum(axis=2) -> (N, k)
        # Bỏ căn bậc hai giúp tăng tốc đáng kể
        dists_sq = np.sum((X[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2, axis=2)
        labels = dists_sq.argmin(axis=1)

        # 3. Cập nhật centroid = trung bình các pixel thuộc cụm
        new_centroids = np.zeros_like(centroids)
        for j in range(k):
            members = X[labels == j]
            if len(members) > 0:
                new_centroids[j] = members.mean(axis=0)
            else:
                # Nếu cụm bị rỗng, gán lại tâm cụm bằng 1 pixel ngẫu nhiên từ X
                new_centroids[j] = X[rng.choice(len(X))]

        # 4. Kiểm tra điều kiện hội tụ (tổng khoảng cách dịch chuyển của các centroids)
        shift = np.linalg.norm(new_centroids - centroids)
        centroids = new_centroids
        
        if shift < tol:
            break

    return centroids, labels