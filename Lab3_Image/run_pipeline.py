import os
import sys
from pathlib import Path

# Tự động thêm thư mục gốc dự án vào sys.path để chạy trực tiếp không bị lỗi Import
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
from src.image_loader import load_image, to_rgb
from src.kmeans import kmeans
from src.preprocessing import flatten, resize_image
from src.segmentation import segment
from src.visualization import show

# ---- Cấu hình mặc định ----
IMAGE_PATH = "data/raw/OIP.webp"
COLOR_SPACE = "RGB"        # "RGB", "HSV" hoặc "LAB"
K_VALUES = [4]             # Thử nhiều giá trị K, ví dụ: [2, 4, 8, 16]
MAX_SIDE = 200             # Cạnh dài nhất sau khi resize


def main():
    # Kiểm tra sự tồn tại của file ảnh đầu vào
    if not os.path.exists(IMAGE_PATH):
        print(f"Lỗi: Không tìm thấy file ảnh tại '{IMAGE_PATH}'.")
        print("Vui lòng kiểm tra lại đường dẫn hoặc thêm ảnh vào thư mục data/raw/")
        return

    # 1. Load ảnh và đổi không gian màu
    img = load_image(IMAGE_PATH, COLOR_SPACE)

    # 2. Tiền xử lý: resize và flatten
    small = resize_image(img, MAX_SIDE)
    X, shape = flatten(small)
    print(f"--- Đã nạp ảnh thành công ---")
    print(f"Kích thước sau resize: {shape[:2]} (H x W) | Số pixel: {len(X)}")

    # Tự động tạo thư mục đầu ra nếu chưa tồn tại
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    for k in K_VALUES:
        print(f"\n[+] Đang phân cụm K-Means với K = {k}...")
        
        # 3. K-Means
        centroids, labels = kmeans(X, k)

        # 4. Phân đoạn: Thay pixel bằng màu centroid & reshape về kích thước ảnh
        seg = segment(centroids, labels, shape)

        # 5. Đổi màu về RGB để hiển thị và lưu ảnh
        original_rgb = to_rgb(small, COLOR_SPACE)
        segmented_rgb = to_rgb(seg, COLOR_SPACE)

        # Hiển thị & lưu biểu đồ so sánh
        show(
            original_rgb,
            segmented_rgb,
            k,
            save_path=f"figures/segmentation_k{k}.png"
        )

        # Lưu ảnh kết quả chất lượng cao (Đổi RGB sang BGR cho OpenCV imwrite)
        cv2.imwrite(
            f"results/segmented_k{k}.png",
            cv2.cvtColor(segmented_rgb, cv2.COLOR_RGB2BGR)
        )
        print(f"--> Đã lưu thành công: 'figures/segmentation_k{k}.png' và 'results/segmented_k{k}.png'")


if __name__ == "__main__":
    main()