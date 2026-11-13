import cv2
import numpy as np


def resize_image(img, max_side=200):
    """Thu nhỏ ảnh (giữ tỉ lệ) để K-Means chạy nhanh hơn."""
    if img is None:
        raise ValueError("Ảnh đầu vào không hợp lệ (None).")
    
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:                      # ảnh đã nhỏ thì giữ nguyên
        return img
    
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)


def flatten(img):
    """Biến đổi ảnh (H, W, C) -> mảng 2D (H*W, C): mỗi pixel là 1 vector đặc trưng float32."""
    if img.ndim == 2:  # Xử lý trường hợp ảnh xám (H, W)
        img = np.expand_dims(img, axis=-1)
        
    h, w, c = img.shape
    X = img.reshape(-1, c).astype(np.float32)
    return X, (h, w, c)

def reconstruct_image(X_flattened, orig_shape):
    """Tái tạo lại ảnh 3D từ mảng 2D pixel đã phân đoạn về dạng uint8."""
    return X_flattened.reshape(orig_shape).clip(0, 255).astype(np.uint8)