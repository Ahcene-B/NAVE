# SPDX-License-Identifier: MIT
"""The engine behind `NAVEImage`: one image, processed whole or in overlapping windows.

* Images up to 1024 x 1024 pixels in total (window="auto") go through the network whole.
* Larger images are cut into overlapping windows (448 px, 32 px margins); only each window's
  centre is kept, where features are not distorted by the window edge.

Composite pixels are produced a few rows at a time; concept scores are computed on the feature
grid and interpolated to pixels, so concept boundaries can be finer than the grid.
"""
import math
from dataclasses import dataclass

import numpy as np

from . import memory
from ._images import normalize as normalize_image, pad
from .clustering import BayesianKMeans
from .composite import CompositeFeatures
from .interp import grid_coords, interp_matrix

AUTO_MAX_PIXELS = 1024 * 1024
AUTO_WINDOW = 448
MARGIN = 32


# --------------------------------------------------------------------------
# Tiling
# --------------------------------------------------------------------------
@dataclass
class Tile:
    iy: int
    ix: int
    y0: int                                 # tile origin in the padded image
    x0: int
    rows: tuple                             # kept pixel rows inside the tile, [start, end)
    cols: tuple
    out_rows: tuple                         # where they go in the image
    out_cols: tuple


class Tiling:
    """Windows over a (H, W) image. One tile covering the padded image means "whole image"."""

    def __init__(self, H, W, tile_h, tile_w, margin):
        self.H, self.W, self.tile_h, self.tile_w, self.margin = H, W, tile_h, tile_w, margin
        self.step_h, self.step_w = tile_h - 2 * margin, tile_w - 2 * margin
        self.ny, self.nx = math.ceil(H / self.step_h), math.ceil(W / self.step_w)

    @classmethod
    def plan(cls, H, W, window, backbone, factor=3):
        """Whole image, or windows of W px every W/2 px.

        window="auto": windows when the image's longer side is at least `factor` times the
        model's input size; they are twice the input size (the input size itself for models
        tied to one size). Every window keeps its central half (margins of about W/4), so each
        pixel comes from the window whose centre is nearest, far from any window edge.
        """
        m = backbone.multiple
        if backbone.fixed_size:
            th, tw = backbone.fixed_size
            if H <= th and W <= tw:
                return cls(H, W, th, tw, 0)
            return cls(H, W, th, tw, cls.margin_for(min(th, tw), m))
        if window == "auto":
            size = getattr(backbone, "input_size", 224)
            window = None if max(H, W) < factor * size else 2 * size
        if window is None:
            return cls(H, W, math.ceil(H / m) * m, math.ceil(W / m) * m, 0)
        window = int(window)
        if window % m or window < 4 * m:
            raise ValueError(f"window must be a multiple of {m}, at least {4 * m} px (got {window}).")
        if H <= window and W <= window:
            return cls(H, W, math.ceil(H / m) * m, math.ceil(W / m) * m, 0)
        return cls(H, W, window, window, cls.margin_for(window, m))

    @staticmethod
    def margin_for(window, multiple):
        """About a quarter of the window, on whole feature cells."""
        return max(multiple, (window // 4) // multiple * multiple)

    @property
    def n(self):
        return self.ny * self.nx

    @property
    def description(self):
        return "whole image" if self.n == 1 else f"{self.ny} x {self.nx} windows of {self.tile_h} px"

    def padded(self, rgb):
        m = self.margin                         # padded size: (n - 1) * step + tile = n * step + 2m
        return pad(rgb, m, self.ny * self.step_h + m - self.H, m, self.nx * self.step_w + m - self.W)

    def tiles(self):
        m = self.margin
        for iy in range(self.ny):
            for ix in range(self.nx):
                y, x = iy * self.step_h, ix * self.step_w
                h, w = min(self.step_h, self.H - y), min(self.step_w, self.W - x)
                yield Tile(iy, ix, y, x, (m, m + h), (m, m + w), (y, y + h), (x, x + w))


# --------------------------------------------------------------------------
# The engine
# --------------------------------------------------------------------------
class _Engine:
    """Runs one segmentation; holds the image, backbone and options."""

    def __init__(self, rgb, backbone, names, tiling, opts, report):
        self.rgb, self.backbone, self.names, self.tiling = rgb, backbone, names, tiling
        self.o, self.report = opts, report
        self.padded = tiling.padded(rgb)
        self._cached = None
        self._checked_memory = False
        self.temperature = None if opts["temperature"] == "auto" else float(opts["temperature"])

    # -- features ----------------------------------------------------------
    def features(self, tile):
        if self._cached is not None:
            return self._cached
        y, x = tile.y0, tile.x0
        crop = self.padded[y:y + self.tiling.tile_h, x:x + self.tiling.tile_w]
        if crop.dtype == np.uint8:                  # default transforms: ImageNet-style normalisation here
            x_in = normalize_image(crop, self.backbone.mean, self.backbone.std)
        else:                                       # already transformed by the user's transforms
            x_in = np.ascontiguousarray(crop.transpose(2, 0, 1)[None], np.float32)
        maps = self.backbone(x_in, self.names)
        if not self._checked_memory:
            self._check_measured_memory(maps)
        comp = CompositeFeatures([maps[n][0] for n in self.names], self.o["upsample"], self.o["normalize"],
                                 self.o["scale"], self.o["block_weights"])
        if self.tiling.n == 1:
            self._cached = comp
        return comp

    def _check_measured_memory(self, maps):
        """For models without a calibrated estimate: estimate from the first window's features."""
        self._checked_memory = True
        if getattr(self.backbone, "name", None) in memory.MODELS:
            return
        feats = memory.array_bytes(*maps.values())
        shallow = maps[self.names[0]]
        cells = shallow.shape[2] * shallow.shape[3]
        dim = sum(m.shape[1] for m in maps.values())
        need = 4 * feats + memory.pipeline_bytes(
            feats, cells, dim, self.tiling.tile_h * self.tiling.tile_w, self.tiling.H * self.tiling.W,
            self.o["k"], self.tiling.n, self.o["two_pass"], self.o["fit_on"] == "all",
            self.o["entropy"], self.o["sample_size"])
        memory.warn_if_too_large(need)

    def grid_range(self, comp, tile):
        """Grid rows and columns covering the tile's kept pixels."""
        sy, sx = self.tiling.tile_h / comp.height, self.tiling.tile_w / comp.width
        return ((int(tile.rows[0] // sy), int(math.ceil(tile.rows[1] / sy))),
                (int(tile.cols[0] // sx), int(math.ceil(tile.cols[1] / sx))))

    def _tiles(self, report, text):
        tiles = list(self.tiling.tiles())
        for i, tile in enumerate(tiles):
            yield tile
            report((i + 1) / len(tiles), f"{text} {i + 1}/{len(tiles)}" if len(tiles) > 1 else text)

    # -- fitting -----------------------------------------------------------
    def sample(self, n, report):
        area = sum((t.rows[1] - t.rows[0]) * (t.cols[1] - t.cols[0]) for t in self.tiling.tiles())
        parts = []
        for tile in self._tiles(report, "sampling window"):
            comp = self.features(tile)
            rows, cols = self.grid_range(comp, tile)
            share = (tile.rows[1] - tile.rows[0]) * (tile.cols[1] - tile.cols[0]) / area
            rng = np.random.default_rng([self.o["seed"], tile.iy, tile.ix])
            parts.append(comp.sample(max(1, int(round(n * share))), rng, rows, cols))
        return np.concatenate(parts)

    # -- assignment --------------------------------------------------------
    def pixel_scores(self, scores, rows, cols):
        """Yield (r0, r1, scores) with scores (r1 - r0, cols, K) interpolated to tile pixels."""
        gh, gw, K = scores.shape
        mode = self.o["upsample"]
        Ah = interp_matrix(grid_coords(np.arange(*rows), gh, self.tiling.tile_h), gh, mode)
        Aw = interp_matrix(grid_coords(np.arange(*cols), gw, self.tiling.tile_w), gw, mode)
        flat = scores.reshape(gh, gw * K)
        step = max(1, (16 * 2 ** 20) // max(1, (cols[1] - cols[0]) * K))
        for r in range(0, rows[1] - rows[0], step):
            t = (Ah[r:r + step] @ flat).reshape(-1, gw, K)
            yield r, r + len(t), np.matmul(Aw, t)

    def resolve_temperature(self, scores, clusterer):
        """temperature="auto": 1 for probabilistic clusterers, else the median gap between the
        best and second-best cluster score of the grid cells (so that gap counts as 1)."""
        if isinstance(clusterer, BayesianKMeans) or getattr(clusterer, "kind", None) == "proba":
            return 1.0
        flat = scores.reshape(-1, scores.shape[2])
        if flat.shape[1] < 2:
            return 1.0
        top2 = np.partition(flat, -2, axis=1)[:, -2:]
        return max(float(np.median(top2[:, 1] - top2[:, 0])), 1e-6)

    def labels_and_entropy(self, scores, rows, cols, clusters_out, entropy_out, T=1.0):
        """Write argmax labels and, if asked, the uncertainty H(softmax(scores / T)) / log K
        (in [0, 1], K = number of concepts) for tile pixels rows x cols."""
        for r0, r1, s in self.pixel_scores(scores, rows, cols):
            clusters_out[r0:r1] = s.argmax(2)
            if entropy_out is not None:
                entropy_out[r0:r1] = normalised_entropy(s, T)

    def assign(self, clusterer, report):
        H, W = self.tiling.H, self.tiling.W
        clusters = np.empty((H, W), np.int32)
        entropy = np.empty((H, W), np.float32) if self.o["entropy"] else None
        for tile in self._tiles(report, "assigning window"):
            comp = self.features(tile)
            scores = comp.scores(clusterer)
            if entropy is not None and self.temperature is None:
                self.temperature = self.resolve_temperature(scores, clusterer)
            (a, b), (c, d) = tile.out_rows, tile.out_cols
            self.labels_and_entropy(scores, tile.rows, tile.cols, clusters[a:b, c:d],
                                    None if entropy is None else entropy[a:b, c:d], self.temperature)
        return clusters, entropy


def normalised_entropy(scores, temperature):
    """H(softmax(scores / T)) / log K along the last axis, clamped to [0, 1].

    scores: (..., K), higher = more likely, e.g. minus squared distances to the prototypes.
    Computed in float64 with a log-softmax; 1 - this value is the certainty."""
    K = scores.shape[-1]
    if K < 2:
        return np.zeros(scores.shape[:-1], np.float32)
    z = scores.astype(np.float64) / max(float(temperature), 1e-30)
    z -= z.max(-1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(-1, keepdims=True))
    H = -(np.exp(logp) * logp).sum(-1)
    return np.clip(H / math.log(K), 0.0, 1.0).astype(np.float32)


def _attr(obj, name):
    value = getattr(obj, name, None)
    return None if value is None else np.asarray(value)


