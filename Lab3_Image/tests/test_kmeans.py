import numpy as np

from src.kmeans import kmeans
from src.segmentation import segment


def test_output_shapes():
    X = np.random.default_rng(0).random((500, 3)).astype(np.float32) * 255
    centroids, labels = kmeans(X, k=4)
    assert centroids.shape == (4, 3)
    assert labels.shape == (500,)
    assert labels.min() >= 0 and labels.max() < 4


def test_two_clear_clusters():
    rng = np.random.default_rng(0)
    dark = rng.normal(20, 2, (100, 3))
    bright = rng.normal(200, 2, (100, 3))
    X = np.vstack([dark, bright]).astype(np.float32)

    centroids, labels = kmeans(X, k=2)

    means = sorted(centroids.mean(axis=1))
    assert abs(means[0] - 20) < 5
    assert abs(means[1] - 200) < 5
    assert len(set(labels[:100])) == 1      # nhóm tối cùng một cụm
    assert len(set(labels[100:])) == 1      # nhóm sáng cùng một cụm
    assert labels[0] != labels[-1]


def test_segment_restores_shape_and_dtype():
    h, w = 6, 8
    X = np.random.default_rng(1).random((h * w, 3)).astype(np.float32) * 255
    centroids, labels = kmeans(X, k=3)
    seg = segment(centroids, labels, (h, w, 3))
    assert seg.shape == (h, w, 3)
    assert seg.dtype == np.uint8
    assert len(np.unique(seg.reshape(-1, 3), axis=0)) <= 3

def test_inertia_decreases_when_k_increases():
    from src.eval import inertia

    X = np.random.default_rng(2).random((300, 3)).astype(np.float32) * 255
    scores = []
    for k in (2, 4, 8):
        centroids, labels = kmeans(X, k)
        scores.append(inertia(X, centroids, labels))
    assert scores[0] > scores[1] > scores[2]