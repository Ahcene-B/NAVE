# SPDX-License-Identifier: MIT
"""Peak-memory estimates and warnings.

Estimates for the built-in models were calibrated against measured runs and err on the
high side. For other models the estimate is made from the first window's real feature
maps, before the rest of the work.
"""
import os
import subprocess
import sys
import warnings

import numpy as np

GB = 2 ** 30
MB = 2 ** 20

# built-in models: block channels, block strides, weights in MB, runtime memory factor
MODELS = {
    "vitt16": ([192] * 12, [16] * 12, 22, 4.0),
    "resnet18": ([64, 128, 256, 512], [4, 8, 16, 32], 47, 4.5),
    "resnet50": ([256, 512, 1024, 2048], [4, 8, 16, 32], 102, 2.5),
    "vits16": ([384] * 12, [16] * 12, 88, 4.0),
}


VIT_HEADS = {"vits16": 6, "vitt16": 3}


class NaveMemoryWarning(UserWarning):
    """The run may need more memory than is currently free."""


def available():
    """Bytes of memory available now, or None if unknown."""
    try:
        if sys.platform.startswith("linux"):
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) * 1024
        elif sys.platform == "darwin":
            out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=10).stdout
            lines = out.splitlines()
            page = int(lines[0].split("page size of")[1].split()[0]) if "page size of" in lines[0] else 4096
            pages = {}
            for line in lines[1:]:
                key, _, val = line.partition(":")
                val = val.strip().rstrip(".")
                if val.isdigit():
                    pages[key.strip()] = int(val)
            return page * sum(pages.get(k, 0) for k in
                              ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable"))
        elif os.name == "nt":
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + \
                           [(n, ctypes.c_ulonglong) for n in ("total", "avail", "total_page", "avail_page",
                                                             "total_virtual", "avail_virtual", "avail_ext")]
            st = MemoryStatus()
            st.dwLength = ctypes.sizeof(MemoryStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
                return int(st.avail)
    except Exception:
        pass
    return None


def network_bytes(model, tile_h, tile_w):
    """Memory of one forward pass of a built-in model: outputs plus runtime working memory."""
    chans, strides, _, factor = MODELS[model]
    outputs = sum(4 * c * (tile_h // s) * (tile_w // s) for c, s in zip(chans, strides))
    total = (1 + factor) * outputs
    if model in VIT_HEADS:
        n = (tile_h // 16) * (tile_w // 16) + 1
        total += 3.0 * 4 * VIT_HEADS[model] * n * n     # attention scores: quadratic in tokens
    return total


def pipeline_bytes(feature_bytes, grid_cells, dim, tile_pixels, image_pixels, k,
                   n_tiles, two_pass, fit_all, entropy, sample_size):
    """Memory of everything around the network, given the size of one tile's features."""
    composite = 3 * 64 * MB + 4 * grid_cells * k + 2 * 128 * MB       # chunks, scores, pixel chunks
    fit = 2 * 4 * min(sample_size, grid_cells * n_tiles) * dim
    if fit_all:
        fit = max(fit, 4 * grid_cells * n_tiles * dim)
    labels = 0 if two_pass or n_tiles == 1 else 48 * tile_pixels
    image = 3 * image_pixels + 60 * image_pixels + (4 * image_pixels if entropy else 0)
    return feature_bytes + composite + fit + labels + image


def estimate(model, block_ids, tile_h, tile_w, n_tiles, image_pixels, k=10, two_pass=False,
             fit_all=False, entropy=False, sample_size=20000):
    """Estimated peak bytes of a run with a built-in model (None for other models)."""
    if model not in MODELS:
        return None
    chans, strides, weights_mb, _ = MODELS[model]
    sel = [i - 1 for i in block_ids]
    finest = min(strides[i] for i in sel)
    cells = (tile_h // finest) * (tile_w // finest)
    dim = sum(chans[i] for i in sel)
    features = sum(4 * chans[i] * (tile_h // strides[i]) * (tile_w // strides[i]) for i in sel)
    rest = pipeline_bytes(features, cells, dim, tile_h * tile_w, image_pixels, k,
                          n_tiles, two_pass, fit_all, entropy, sample_size)
    return int(3 * weights_mb * MB + 150 * MB + network_bytes(model, tile_h, tile_w) + rest)


def warn_if_too_large(need, advice=""):
    """Warn (NaveMemoryWarning) when `need` bytes may not fit in the free memory."""
    free = available()
    if need is None or free is None or need < 0.8 * free:
        return
    warnings.warn(f"This run may need about {need / GB:.1f} GB of memory and about {free / GB:.1f} GB is free. "
                  f"It may be very slow or fail.{(' ' + advice) if advice else ''}",
                  NaveMemoryWarning, stacklevel=3)


def advice(model, blocks, window):
    tips = []
    if model in VIT_HEADS:
        tips.append("use smaller windows (ViT memory grows with the square of the window area)")
    elif window and window > 448:
        tips.append("use smaller windows")
    if model in ("resnet18", "resnet50") and 1 in blocks:
        tips.append("leave out block 1 (the largest)")
    return ("To use less memory: " + "; ".join(tips) + ".") if tips else ""


def array_bytes(*arrays):
    return int(sum(np.asarray(a).nbytes for a in arrays))
