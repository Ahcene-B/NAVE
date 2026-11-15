# SPDX-License-Identifier: MIT
"""Separable interpolation written as small matrices.

Resampling an (h, w, c) map to (H, W, c) is `Ah @ X @ Aw.T` per channel, where Ah is
(H, h) and Aw is (W, w). Rows of these matrices can be taken for any subset of output
positions, which lets callers produce large outputs a few rows at a time.
"""
import math

import numpy as np

MODES = ("nearest", "linear", "cubic")


def check_mode(mode):
    if mode not in MODES:
        raise ValueError(f"upsample must be one of {MODES}, got {mode!r}.")
    return mode


def resize_coords(n_in, n_out):
    """Source coordinates of `n_out` evenly spread samples (pixel centres, align_corners=False)."""
    return (np.arange(n_out) + 0.5) * (n_in / n_out) - 0.5


def _keys(d, a=-0.75):
    """Keys cubic kernel, as used by PyTorch's bicubic interpolation."""
    d = np.abs(d)
    return np.where(d <= 1, ((a + 2) * d - (a + 3)) * d * d + 1,
                    np.where(d < 2, ((a * d - 5 * a) * d + 8 * a) * d - 4 * a, 0.0))


def interp_matrix(coords, n_in, mode):
    """Matrix of shape (len(coords), n_in) that samples a length-`n_in` signal at `coords`."""
    check_mode(mode)
    coords = np.asarray(coords, np.float64)
    rows = np.arange(len(coords))
    A = np.zeros((len(coords), n_in), np.float32)
    if mode == "nearest":
        A[rows, np.clip(np.floor(coords + 0.5).astype(int), 0, n_in - 1)] = 1.0
    elif mode == "linear":
        x = np.clip(coords, 0, n_in - 1)
        i0 = np.floor(x).astype(int)
        t = x - i0
        np.add.at(A, (rows, i0), 1 - t)
        np.add.at(A, (rows, np.minimum(i0 + 1, n_in - 1)), t)
    else:
        i0 = np.floor(coords).astype(int)
        t = coords - i0
        for k in (-1, 0, 1, 2):
            np.add.at(A, (rows, np.clip(i0 + k, 0, n_in - 1)), _keys(t - k))
    return A


def area_matrix(n_in, n_out):
    """Box-filter matrix for shrinking a signal from `n_in` to `n_out` samples."""
    edges = np.arange(n_out + 1) * (n_in / n_out)
    A = np.zeros((n_out, n_in), np.float32)
    for o in range(n_out):
        a, b = edges[o], edges[o + 1]
        for i in range(int(math.floor(a)), min(int(math.ceil(b)), n_in)):
            A[o, i] = min(b, i + 1) - max(a, i)
        A[o] /= A[o].sum()
    return A


def apply_separable(Ah, Aw, X):
    """Resample X (h, w, c) with row matrix Ah (H, h) and column matrix Aw (W, w)."""
    h, w, c = X.shape
    Y = (Ah @ X.reshape(h, w * c)).reshape(Ah.shape[0], w, c)
    return np.matmul(Aw, Y)


def resample(X, height, width, mode):
    """Resample a (h, w) or (h, w, c) map to (height, width[, c])."""
    squeeze = X.ndim == 2
    X = X[..., None] if squeeze else X
    h, w = X.shape[:2]
    Ah = interp_matrix(resize_coords(h, height), h, mode)
    Aw = interp_matrix(resize_coords(w, width), w, mode)
    Y = apply_separable(Ah, Aw, X.astype(np.float32, copy=False))
    return Y[..., 0] if squeeze else Y


def grid_coords(pixels, grid_size, tile_size):
    """Grid coordinates of pixel centres `pixels` in a tile of `tile_size` px covered by
    `grid_size` cells (each cell is tile_size / grid_size px)."""
    return (np.asarray(pixels, np.float64) + 0.5) * grid_size / tile_size - 0.5
