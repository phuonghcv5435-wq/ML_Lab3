import argparse
import sys
from pathlib import Path

# 1. Tự động thêm thư mục gốc vào sys.path để bấm nút RUN trực tiếp không bị lỗi Import
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib.pyplot as plt
import numpy as np

from src.image_loader import load_image
from src.kmeans import kmeans
from src.preprocessing import flatten, resize_image


def run_demo(image_path, k=4, max_side=300, color_space="RGB"):
    print(f"--- Đang chạy Demo K-Means Image Segmentation ---")
    print(f"Ảnh: {image_path} | Số cụm K = {k} | Không gian màu: {color_space}")

    # Đọc ảnh
    img_orig = load_image(image_path, color_space=color_space)

    # Tiền xử lý
    img_resized = resize_image(img_orig, max_side=max_side)
    X, orig_shape = flatten(img_resized)

    # Phân cụm K-Means
    centroids, labels = kmeans(X, k=k)

    # Tái tạo ảnh 3D
    X_segmented = centroids[labels]
    img_segmented = X_segmented.reshape(orig_shape).astype(np.uint8)

    # Hiển thị kết quả
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(img_resized)
    axes[0].set_title(f"Ảnh gốc ({orig_shape[1]}x{orig_shape[0]})")
    axes[0].axis("off")

    axes[1].imshow(img_segmented)
    axes[1].set_title(f"Ảnh phân đoạn (K = {k})")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 2. Tự động tìm file ảnh đầu tiên trong data/raw nếu chạy trực tiếp
    raw_dir = ROOT_DIR / "data" / "raw"
    valid_exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    
    default_img = None
    if raw_dir.exists():
        found_imgs = [f for f in raw_dir.iterdir() if f.suffix.lower() in valid_exts]
        if found_imgs:
            default_img = str(found_imgs[0])  # Sẽ tự chọn OIP.webp

    if not default_img:
        default_img = "data/raw/OIP.webp"

    parser = argparse.ArgumentParser(description="K-Means Image Segmentation Demo")
    parser.add_argument("--image", type=str, default=default_img, help="Đường dẫn ảnh")
    parser.add_argument("--k", type=int, default=5, help="Số lượng cụm màu (K)")
    args = parser.parse_args()

    run_demo(args.image, k=args.k)