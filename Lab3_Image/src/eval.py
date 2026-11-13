import os
import sys
from pathlib import Path

# Thêm tự động thư mục gốc của dự án vào sys.path để chạy trực tiếp không bị lỗi Import
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib.pyplot as plt
import numpy as np

from src.kmeans import kmeans


def inertia(X, centroids, labels):
    """Tính tổng bình phương khoảng cách từ mỗi pixel đến centroid của cụm tương ứng (WCSS/Inertia)."""
    return float(((X - centroids[labels]) ** 2).sum())


def elbow(X, k_values, save_path="figures/elbow.png", display=False):
    """Chạy K-Means với nhiều giá trị K, vẽ đồ thị Inertia theo K (phương pháp Elbow)."""
    scores = []
    print("\n--- Đang chạy phương pháp Elbow ---")
    for k in k_values:
        centroids, labels = kmeans(X, k)
        val = inertia(X, centroids, labels)
        scores.append(val)
        print(f"K = {k:>2}: Inertia = {val:,.1f}")

    fig = plt.figure(figsize=(6, 4))
    plt.plot(k_values, scores, marker="o", color="b", linestyle="--")
    plt.xlabel("Số lượng cụm (K)")
    plt.ylabel("Inertia (WCSS)")
    plt.title("Elbow Method for Optimal K")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        dir_name = os.path.dirname(save_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"--> Đã lưu biểu đồ Elbow tại '{save_path}'")

    if display:
        plt.show()

    plt.close(fig)
    return scores


def compare_with_sklearn(X, k=4):
    """So sánh Inertia giữa bản tự cài đặt và scikit-learn KMeans."""
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        print("\n[!] Chưa cài scikit-learn, bỏ qua phần so sánh (pip install scikit-learn).")
        return None

    print(f"\n--- So sánh kết quả với scikit-learn (K = {k}) ---")
    
    # 1. Thuật toán tự cài đặt
    centroids, labels = kmeans(X, k)
    ours = inertia(X, centroids, labels)

    # 2. Thuật toán từ scikit-learn
    sk_model = KMeans(n_clusters=k, n_init=10, random_state=42)
    sk_model.fit(X)
    theirs = float(sk_model.inertia_)

    diff_percent = abs(ours - theirs) / theirs * 100
    print(f"Tự cài đặt : Inertia = {ours:,.1f}")
    print(f"Scikit-learn: Inertia = {theirs:,.1f}")
    print(f"Chênh lệch  : {diff_percent:.2f}%")

    return ours, theirs


if __name__ == "__main__":
    from run_pipeline import COLOR_SPACE, IMAGE_PATH, MAX_SIDE
    from src.image_loader import load_image
    from src.preprocessing import flatten, resize_image

    if os.path.exists(IMAGE_PATH):
        img = load_image(IMAGE_PATH, COLOR_SPACE)
        small = resize_image(img, MAX_SIDE)
        X, _ = flatten(small)

        elbow(X, [2, 3, 4, 5, 6, 8, 10, 12, 16], save_path="figures/elbow.png")
        compare_with_sklearn(X, k=4)
    else:
        print(f"Lỗi: Không tìm thấy file ảnh tại '{IMAGE_PATH}' để chạy eval.py.")