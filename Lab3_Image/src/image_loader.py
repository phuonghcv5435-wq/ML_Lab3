import cv2
import numpy as np
from PIL import Image

# OpenCV đọc ảnh ở dạng BGR nên cần đổi sang không gian màu mong muốn
FROM_BGR = {
    "RGB": cv2.COLOR_BGR2RGB,
    "HSV": cv2.COLOR_BGR2HSV,
    "LAB": cv2.COLOR_BGR2LAB,
}
TO_RGB = {
    "RGB": None,
    "HSV": cv2.COLOR_HSV2RGB,
    "LAB": cv2.COLOR_LAB2RGB,
}


def load_image(path, color_space="RGB"):
    """Đọc ảnh bằng OpenCV (hoặc PIL fallback) và đổi sang không gian màu color_space."""
    if color_space not in FROM_BGR:
        raise ValueError(f"color_space phải là một trong {list(FROM_BGR)}")

    img = None
    # 1. Thử đọc bằng OpenCV (np.fromfile giúp hỗ trợ đường dẫn tiếng Việt)
    try:
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        img = None

    # 2. Nếu OpenCV không đọc được (ví dụ file .webp bị lỗi codec), dùng PIL để hỗ trợ
    if img is None:
        try:
            pil_img = Image.open(path).convert("RGB")
            # Chuyển PIL (RGB) sang OpenCV (BGR) để giữ nguyên quy trình cvtColor phía dưới
            img_rgb = np.array(pil_img)

            # Nếu mục tiêu là RGB thì trả về ngay không cần qua BGR
            if color_space == "RGB":
                return img_rgb

            # Nếu mục tiêu là HSV hoặc LAB, đổi sang BGR rồi mới đổi tiếp bằng OpenCV
            img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        except Exception as e:
            raise FileNotFoundError(f"Không đọc được ảnh: {path}. Lỗi: {e}")

    return cv2.cvtColor(img, FROM_BGR[color_space])


def to_rgb(img, color_space="RGB"):
    """Đổi ảnh từ không gian màu bất kỳ về lại RGB để hiển thị bằng Matplotlib."""
    if color_space not in TO_RGB:
        raise ValueError(f"color_space không hợp lệ! Chọn một trong: {list(TO_RGB)}")
        
    code = TO_RGB[color_space]
    return img if code is None else cv2.cvtColor(img, code)