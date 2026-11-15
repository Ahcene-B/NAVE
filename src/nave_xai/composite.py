# SPDX-License-Identifier: MIT
"""Composite pixels: Algorithm 1 of the NAVE paper, lines 6-11.

Each selected block's feature map is upsampled to the grid of the shallowest selected
block, L2-normalised per position (optional), scaled by C_min / (1 + C_b) (optional),
weighted (optional) and concatenated. Composite pixels are produced a few rows at a time,
so a large window never needs all of them in memory at once.
"""
import numpy as np

from .interp import interp_matrix, resize_coords

CHUNK_FLOATS = 16 * 2 ** 20          # about 64 MB of composite pixels per chunk


class CompositeFeatures:
    """Composite pixels of one image (or window).

    Parameters
    ----------
    maps : list of (C, h, w) arrays, shallowest block first (it sets the grid)
    mode : "nearest", "linear" or "cubic", to upsample deeper blocks
    normalize, scale : the two steps of Algorithm 1 (lines 9 and 10)
    weights : optional extra weight per block
    """

    def __init__(self, maps, mode="cubic", normalize=True, scale=True, weights=None):
        if weights is not None and len(weights) != len(maps):
            raise ValueError(f"block_weights has {len(weights)} values for {len(maps)} blocks.")
        grids = [m.shape[1:] for m in maps]
        order = np.argsort([-(h * w) for h, w in grids], kind="stable")
        ref = int(order[0])                                   # finest grid = shallowest block
        self.height, self.width = grids[ref]
        c_ref = maps[ref].shape[0]
        self.normalize = normalize
        self._parts = []
        for i, m in enumerate(maps):
            C, h, w = m.shape
            factor = (c_ref / (1.0 + C) if scale else 1.0) * (1.0 if weights is None else float(weights[i]))
            flat = np.ascontiguousarray(m.transpose(1, 2, 0), np.float32).reshape(h, w * C)
            rows = None if h == self.height else interp_matrix(resize_coords(h, self.height), h, mode)
            cols = None if w == self.width else interp_matrix(resize_coords(w, self.width), w, mode)
            self._parts.append((flat, C, w, rows, cols, np.float32(factor)))
        self.dim = sum(p[1] for p in self._parts)
        self.chunk_rows = max(1, CHUNK_FLOATS // max(1, self.width * self.dim))

    @property
    def shape(self):
        return self.height, self.width, self.dim

    def rows(self, r0, r1):
        """Composite pixels of grid rows [r0, r1): array (r1 - r0, width, dim)."""
        out = []
        for flat, C, w, rows, cols, factor in self._parts:
            x = (flat[r0:r1] if rows is None else rows[r0:r1] @ flat).reshape(r1 - r0, w, C)
            if cols is not None:
                x = np.matmul(cols, x)
            if self.normalize:
                x = x / (np.linalg.norm(x, axis=2, keepdims=True) + 1e-8)
            out.append(x * factor)
        return np.concatenate(out, axis=2)

    def chunks(self, r0=0, r1=None):
        """Yield (start_row, end_row, rows) over grid rows [r0, r1)."""
        r1 = self.height if r1 is None else r1
        for a in range(r0, r1, self.chunk_rows):
            b = min(r1, a + self.chunk_rows)
            yield a, b, self.rows(a, b)

    def sample(self, n, rng, rows=None, cols=None):
        """`n` random composite pixels from the region rows x cols (default: whole grid)."""
        r0, r1 = rows or (0, self.height)
        c0, c1 = cols or (0, self.width)
        cells = (r1 - r0) * (c1 - c0)
        if cells <= 0 or n <= 0:
            return np.zeros((0, self.dim), np.float32)
        idx = np.sort(rng.choice(cells, min(n, cells), replace=False))
        out = []
        for a, b, x in self.chunks(r0, r1):
            local = idx[(idx >= (a - r0) * (c1 - c0)) & (idx < (b - r0) * (c1 - c0))] - (a - r0) * (c1 - c0)
            out.append(x[:, c0:c1].reshape(-1, self.dim)[local])
        return np.concatenate(out)

    def region(self, rows=None, cols=None):
        """Yield chunks (n_cells, dim) of every composite pixel in rows x cols."""
        r0, r1 = rows or (0, self.height)
        c0, c1 = cols or (0, self.width)
        for _, _, x in self.chunks(r0, r1):
            yield x[:, c0:c1].reshape(-1, self.dim)

    def scores(self, clusterer):
        """Cluster scores of every grid cell, higher = closer: array (height, width, K)."""
        out = None
        for a, b, x in self.chunks():
            s = clusterer.scores(x.reshape(-1, self.dim)).reshape(b - a, self.width, -1)
            if out is None:
                out = np.empty((self.height, self.width, s.shape[2]), np.float32)
            out[a:b] = s
        return out

    def full(self):
        """All composite pixels at once: (height, width, dim). Mind the memory."""
        return np.concatenate([x for _, _, x in self.chunks()], axis=0)
