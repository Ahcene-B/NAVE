# SPDX-License-Identifier: MIT
"""
Rough peak-memory estimate for a NAVE run, and the memory currently available.
Standard library only (runs inside GIMP). The figures were calibrated by measuring
onnxruntime and the worker; they aim to err on the high side.
"""
import math
import os
import subprocess
import sys

GB = 2 ** 30
MB = 2 ** 20
MARGIN = 32
SAMPLE = 20000

# per model: (block channels, block strides, weights in MB, network memory factor)
MODELS = {
    "resnet18": ([64, 128, 256, 512], [4, 8, 16, 32], 47, 4.5),
    "resnet50": ([256, 512, 1024, 2048], [4, 8, 16, 32], 102, 2.5),
    "vits16": ([384] * 12, [16] * 12, 88, 4.0),
}


def estimate(model, blocks, window, width, height, clusters, two_pass, n_layers):
    """Estimated peak bytes of the worker plus the plugin's own buffers."""
    chans, strides, weights_mb, factor = MODELS[model]
    # network: all block outputs of one window, plus the runtime's working memory
    outputs = sum(4 * c * (window // s) ** 2 for c, s in zip(chans, strides))
    network = factor * outputs
    if model == "vits16":
        n = (window // 16) ** 2 + 1
        network += 3.0 * 4 * 6 * n * n                 # attention scores, 6 heads, quadratic in tokens
    # NAVE composite for one window
    sel = [b - 1 for b in blocks]
    grid = window // strides[min(sel)]
    dim = sum(chans[i] for i in sel)
    features = sum(4 * chans[i] * (window // strides[i]) ** 2 for i in sel)
    composite = 3 * 64 * MB + 4 * grid * grid * clusters + 2 * 128 * MB
    kmeans = 2 * 4 * min(SAMPLE, grid * grid if not two_pass else SAMPLE) * dim
    window_labels = 0 if two_pass else 48 * window * window
    per_window = outputs + network + features + composite + kmeans + window_labels
    # whole image
    step = window - 2 * MARGIN
    padded = (math.ceil(height / step) * step + 2 * MARGIN) * (math.ceil(width / step) * step + 2 * MARGIN)
    image = 36 * padded + 100 * width * height + (4 * width * height if two_pass else 0)
    layers = (3 + 4 * n_layers) * width * height        # buffers passed to and from GIMP, both sides
    load = 3 * weights_mb * MB + 150 * MB
    return int(load + per_window + image + 2 * layers)


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
            page = 4096
            first = out.splitlines()[0]
            if "page size of" in first:
                page = int(first.split("page size of")[1].split()[0])
            pages = {}
            for line in out.splitlines()[1:]:
                key, _, val = line.partition(":")
                if val.strip().rstrip(".").isdigit():
                    pages[key.strip()] = int(val.strip().rstrip("."))
            free = sum(pages.get(k, 0) for k in ("Pages free", "Pages inactive", "Pages speculative",
                                                   "Pages purgeable"))
            return free * page
        elif os.name == "nt":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            st = MEMORYSTATUSEX()
            st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
                return int(st.ullAvailPhys)
    except Exception:
        pass
    return None


def warning(model, blocks, window, width, height, clusters, two_pass, n_layers):
    """A message if the run may not fit in the free memory, else None."""
    need = estimate(model, blocks, window, width, height, clusters, two_pass, n_layers)
    free = available()
    if free is None or need < 0.8 * free:
        return None
    tips = []
    if model == "vits16" and window > 640:
        tips.append("a smaller window (ViT memory grows very fast with window size)")
    elif window > 448:
        tips.append("a smaller window")
    if 1 in blocks and model != "vits16":
        tips.append("leaving out block 1 (the largest)")
    if not tips:
        tips.append("closing other programs")
    return (f"This run may need about {need / GB:.1f} GB of memory, and about {free / GB:.1f} GB is free now.\n"
            f"Your computer may slow down a lot, or the run may fail.\n\nTo use less, try {' or '.join(tips)}.")
