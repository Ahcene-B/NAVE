# SPDX-License-Identifier: MIT
"""Image input handling: conversion to RGB uint8, resizing, padding."""
import numpy as np

from .interp import area_matrix, interp_matrix, resize_coords, apply_separable


def as_rgb(image):
    """Return `image` as a (H, W, 3) uint8 array.

    Accepts numpy arrays (H, W), (H, W, 3) or (H, W, 4), uint8 or float in [0, 1],
    and PIL images. Alpha is dropped, grayscale is repeated over three channels.
    """
    a = np.asarray(image)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=2)
    if a.ndim != 3 or a.shape[2] not in (3, 4):
        raise ValueError(f"Expected an image of shape (H, W), (H, W, 3) or (H, W, 4), got {a.shape}.")
    a = a[..., :3]
    if a.dtype != np.uint8:
        a = np.asarray(a, np.float64)
        if a.max() > 1.0 + 1e-6:
            raise ValueError("Float images must have values in [0, 1]; pass uint8 for 0-255 data.")
        a = np.round(np.clip(a, 0.0, 1.0) * 255.0).astype(np.uint8)
    return np.ascontiguousarray(a)


def resize(image, height, width):
    """Resize an RGB uint8 image: box filter when shrinking, linear when enlarging."""
    H, W, _ = image.shape
    if (H, W) == (height, width):
        return image
    Ah = area_matrix(H, height) if height < H else interp_matrix(resize_coords(H, height), H, "linear")
    Aw = area_matrix(W, width) if width < W else interp_matrix(resize_coords(W, width), W, "linear")
    out = apply_separable(Ah, Aw, image.astype(np.float32))
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def pad(image, top, bottom, left, right):
    """Reflect-pad an (H, W, C) array (edge padding when the pad exceeds the image)."""
    widths = ((top, bottom), (left, right), (0, 0))
    try:
        return np.pad(image, widths, mode="reflect")
    except ValueError:
        return np.pad(image, widths, mode="edge")


def normalize(tile, mean, std):
    """uint8 (h, w, 3) -> float32 (1, 3, h, w), normalized with `mean` and `std`."""
    x = (tile.astype(np.float32) / 255.0 - np.asarray(mean, np.float32)) / np.asarray(std, np.float32)
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])
