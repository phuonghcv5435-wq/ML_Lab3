import os
import matplotlib.pyplot as plt


def show(original_rgb, segmented_rgb, k, save_path=None, display=True):
    """Hiển thị ảnh gốc và ảnh phân đoạn cạnh nhau (Side-by-side)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # 1. Vẽ ảnh gốc
    axes[0].imshow(original_rgb)
    axes[0].set_title("Original Image", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    # 2. Vẽ ảnh sau phân đoạn K-Means
    axes[1].imshow(segmented_rgb)
    axes[1].set_title(f"Segmented (K = {k})", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    plt.tight_layout()

    # 3. Lưu ảnh biểu đồ vào đĩa nếu có khai báo save_path
    if save_path:
        dir_name = os.path.dirname(save_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    # 4. Hiển thị cửa sổ matplotlib
    if display:
        plt.show()

    # 5. Đóng figure để giải phóng tài nguyên bộ nhớ
    plt.close(fig)