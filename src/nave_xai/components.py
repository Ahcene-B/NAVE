# SPDX-License-Identifier: MIT
"""Connected-component labelling with numpy only.

The label map is split into horizontal runs of equal value. Runs in neighbouring rows
that touch and share a value are linked, and the links are resolved with vectorised
label propagation. Memory is proportional to the number of runs, not of pixels.
"""
import numpy as np


def resolve(n, u, v):
    """Join node pairs (u[i], v[i]); return the smallest node id of each node's group."""
    lab = np.arange(n)
    while len(u):
        m = np.minimum(lab[u], lab[v])
        np.minimum.at(lab, u, m)
        np.minimum.at(lab, v, m)
        nxt = lab[lab]
        while not np.array_equal(nxt, lab):          # pointer jumping to the roots
            lab, nxt = nxt, nxt[nxt]
        if np.array_equal(lab[u], lab[v]):
            break
    return lab


def connected_components(labels, connectivity=8):
    """Label connected regions of equal value.

    Parameters
    ----------
    labels : (H, W) integer array
    connectivity : 4 or 8

    Returns
    -------
    segments : (H, W) int32 array with ids 0 .. n-1
    n : number of segments
    """
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8.")
    H, W = labels.shape
    if H == 0 or W == 0:
        return np.zeros((H, W), np.int32), 0
    start = np.ones((H, W), bool)
    start[:, 1:] = labels[:, 1:] != labels[:, :-1]
    rows, starts = np.nonzero(start)
    ends = np.empty_like(starts)
    ends[:-1] = starts[1:]
    ends[np.r_[rows[1:] != rows[:-1], True]] = W      # last run of each row ends at W
    values = labels[rows, starts]
    first = np.searchsorted(rows, np.arange(H + 1))   # index of each row's first run
    reach = 0 if connectivity == 8 else 1             # 8-connectivity also links diagonals

    us, vs = [], []
    for r in range(1, H):
        a0, a1, b1 = first[r - 1], first[r], first[r + 1]
        As, Ae, Bs, Be = starts[a0:a1], ends[a0:a1], starts[a1:b1], ends[a1:b1]
        lo = np.searchsorted(Ae, Bs + reach, "left")     # runs above ending at or after B's start
        hi = np.searchsorted(As, Be - reach, "right")    # ... and starting at or before B's end
        count = np.maximum(hi - lo, 0)
        total = int(count.sum())
        if not total:
            continue
        b = np.repeat(np.arange(len(Bs)), count)
        a = np.repeat(lo, count) + np.arange(total) - np.repeat(np.cumsum(count) - count, count)
        same = values[a0 + a] == values[a1 + b]
        us.append(a0 + a[same])
        vs.append(a1 + b[same])

    u = np.concatenate(us) if us else np.zeros(0, np.int64)
    v = np.concatenate(vs) if vs else np.zeros(0, np.int64)
    root = resolve(len(starts), u, v)
    _, run_label = np.unique(root, return_inverse=True)
    segments = np.repeat(run_label.astype(np.int32), ends - starts).reshape(H, W)
    return segments, int(run_label.max()) + 1


def segment_means(values, segments, n=None):
    """Replace every value by the mean over its segment."""
    n = int(segments.max()) + 1 if n is None else n
    flat = segments.ravel()
    counts = np.maximum(np.bincount(flat, minlength=n), 1)
    sums = np.bincount(flat, weights=values.ravel().astype(np.float64), minlength=n)
    return (sums / counts)[segments].astype(np.float32)
