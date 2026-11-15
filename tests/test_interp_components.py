from collections import deque

import numpy as np
import pytest

from nave_xai.components import connected_components, segment_means
from nave_xai.interp import interp_matrix, resample, resize_coords


@pytest.mark.parametrize("mode", ["nearest", "linear", "cubic"])
def test_identity_and_constants(mode, rng):
    x = rng.random((7, 5)).astype(np.float32)
    assert np.allclose(resample(x, 7, 5, mode), x, atol=1e-6)
    A = interp_matrix(resize_coords(6, 17), 6, mode)
    assert np.allclose(A.sum(1), 1.0, atol=1e-6)              # constants stay constant


def test_nearest_upsampling_repeats():
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    assert np.array_equal(resample(x, 4, 6, "nearest"), np.repeat(np.repeat(x, 2, 0), 2, 1))


def _flood(img, connectivity):
    H, W = img.shape
    lab = -np.ones((H, W), int)
    steps = [(-1, 0), (1, 0), (0, -1), (0, 1)] + ([(-1, -1), (-1, 1), (1, -1), (1, 1)] if connectivity == 8 else [])
    n = 0
    for y in range(H):
        for x in range(W):
            if lab[y, x] >= 0:
                continue
            q = deque([(y, x)])
            lab[y, x] = n
            while q:
                a, b = q.popleft()
                for dy, dx in steps:
                    c, d = a + dy, b + dx
                    if 0 <= c < H and 0 <= d < W and lab[c, d] < 0 and img[c, d] == img[a, b]:
                        lab[c, d] = n
                        q.append((c, d))
            n += 1
    return lab, n


@pytest.mark.parametrize("connectivity", [4, 8])
def test_components_match_flood_fill(connectivity, rng):
    for _ in range(120):
        H, W = rng.integers(1, 20, 2)
        img = rng.integers(0, rng.integers(1, 4) + 1, (H, W))
        a, na = connected_components(img, connectivity)
        b, nb = _flood(img, connectivity)
        assert na == nb
        assert len(set(zip(a.ravel(), b.ravel()))) == na       # same partition


def test_segment_means():
    seg = np.array([[0, 0, 1], [1, 1, 2]])
    v = np.array([[1.0, 3.0, 4.0], [6.0, 8.0, 5.0]])
    assert np.allclose(segment_means(v, seg), [[2, 2, 6], [6, 6, 5]])
