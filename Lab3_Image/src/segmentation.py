import numpy as np


def segment(centroids, labels, shape):
    """Thay mỗi pixel bằng màu centroid của cụm, rồi reshape về (H, W, C)."""
    seg = centroids[labels]
    seg = np.clip(seg, 0, 255).astype(np.uint8)
    return seg.reshape(shape)