# SPDX-License-Identifier: MIT
"""Concept patches, correlations with classes, and colour families (used by `NAVEImage`).

A *cell* is a connected region of an explanation map assigned to one concept, in one image.
For every concept, the tracker keeps the **closest** cell (smallest mean distance of its
composite pixels to the prototype) and the **largest** one, as small crops, so memory does
not grow with the number of images. Patches are 32 x 32 renderings of these cells.
"""
import colorsys
import warnings
from dataclasses import dataclass

import numpy as np

from ._images import resize

CROP_SIDE = 192            # stored crops are squares of at most this many pixels
PATCH_SIDE = 32
KINDS = ("closest", "largest")


@dataclass
class Cell:
    image: int
    concept: int
    area: int                  # in grid cells
    distance: float            # mean distance of its composite pixels to the concept's prototype
    box: tuple                 # (top, left, bottom, right) in the original image's pixels
    crop: np.ndarray           # (s, s, 3) uint8, a square view centred on the cell, with context
    mask: np.ndarray           # (s, s) bool, the cell inside the crop


def correlation(proportions, labels, classes):
    """(K, C) Pearson correlation between concept proportions (N, K) and the class indicators."""
    P = proportions.astype(np.float64)
    Y = (np.asarray(labels)[:, None] == np.asarray(classes)[None, :]).astype(np.float64)
    P, Y = P - P.mean(0), Y - Y.mean(0)
    denom = np.sqrt((P ** 2).sum(0))[:, None] * np.sqrt((Y ** 2).sum(0))[None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        R = (P.T @ Y) / denom
    return np.nan_to_num(R).astype(np.float32)


def classes_by_frequency(labels):
    ids, counts = np.unique(np.asarray(labels), return_counts=True)
    return ids[np.argsort(-counts, kind="stable")]


def staircase_order(R, threshold=0.1):
    """Rows (concepts) grouped by their best-correlated column, columns in order, strongest
    first within a column; concepts whose best correlation is below `threshold` go last."""
    best = R.argmax(1)
    strength = R.max(1)
    unassigned = strength < threshold
    key_group = np.where(unassigned, R.shape[1], best)
    return np.lexsort((-strength, key_group))


def prediction_groups(R, threshold=0.1):
    """For every concept, the column (class position) it is grouped under, or -1 (unassigned)."""
    best = R.argmax(1)
    return np.where(R.max(1) >= threshold, best, -1)


# --------------------------------------------------------------------------
# Colours: one family per predicted class
# --------------------------------------------------------------------------
_BASE = ["1f77b4", "ff7f0e", "2ca02c", "d62728", "9467bd", "8c564b", "e377c2", "bcbd22", "17becf",
         "aec7e8", "ffbb78", "98df8a", "ff9896", "c5b0d5", "c49c94", "f7b6d2", "dbdb8d", "9edae5"]


def family_palette(groups, vary="lightness", spread=0.5):
    """Colours for concepts already ordered by group: every group gets a base colour, and its
    concepts vary around it in lightness, saturation or hue. Group -1 (unassigned) is grey.

    groups : (K,) group of every concept, in the concepts' order
    vary : "lightness", "saturation", "hue", or a list of these (varied together)
    spread : how far the variation goes, 0 (all identical) to 1
    """
    varies = [vary] if isinstance(vary, str) else list(vary)
    if not varies or any(v not in ("lightness", "saturation", "hue") for v in varies):
        raise ValueError("color_vary must be 'lightness', 'saturation', 'hue' or a list of these.")
    groups = np.asarray(groups)
    colours = np.zeros((len(groups), 3), np.uint8)
    labels = [g for g in dict.fromkeys(groups.tolist()) if g >= 0]
    for g in list(labels) + [-1]:
        members = np.nonzero(groups == g)[0]
        if not len(members):
            continue
        if g < 0:
            base = (0.5, 0.5, 0.5)
        else:
            h = _BASE[labels.index(g) % len(_BASE)]
            base = tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
        hue, light, sat = colorsys.rgb_to_hls(*base)
        steps = np.linspace(-0.5, 0.5, len(members)) * spread if len(members) > 1 else np.zeros(1)
        for m, t in zip(members, steps):
            hh, ll, ss = hue, light, sat
            if "lightness" in varies:
                ll = np.clip(light + 0.6 * t, 0.15, 0.85)
            if "saturation" in varies:
                ss = np.clip(sat + 1.2 * t, 0.1, 1.0)
            if "hue" in varies:
                hh = (hue + 0.12 * t) % 1.0
            colours[m] = np.round(np.array(colorsys.hls_to_rgb(hh, ll, ss)) * 255)
    return colours


class CellTracker:
    """Keeps, for every concept, the closest and the largest cell seen so far.

    min_size : minimum height and width, in pixels, of a cell to be the closest one
        (None: at least two grid cells)
    """

    def __init__(self, k, min_size=None, context=0.2):
        self.k, self.min_size, self.context = k, min_size, context
        self.cells = [dict.fromkeys(KINDS) for _ in range(k)]

    def update(self, index, rgb, concepts, distance, cell_size):
        """concepts: (r, c) concept map on the grid; distance: (r, c) distance of every grid
        cell to its concept's prototype; cell_size: (height, width) of a grid cell in pixels."""
        from .components import connected_components
        regions, n = connected_components(concepts)
        flat = regions.ravel()
        area = np.bincount(flat, minlength=n)
        mean_d = np.bincount(flat, weights=distance.ravel().astype(np.float64), minlength=n) / np.maximum(area, 1)
        concept_of = np.zeros(n, int)
        concept_of[flat] = concepts.ravel()
        ys, xs = np.divmod(np.arange(regions.size), regions.shape[1])
        top = np.full(n, np.inf); bottom = np.full(n, -np.inf)
        left = np.full(n, np.inf); right = np.full(n, -np.inf)
        np.minimum.at(top, flat, ys); np.maximum.at(bottom, flat, ys)
        np.minimum.at(left, flat, xs); np.maximum.at(right, flat, xs)
        height = (bottom - top + 1) * cell_size[0]
        width = (right - left + 1) * cell_size[1]
        if self.min_size is None:
            eligible = area >= 2
        else:
            eligible = (height >= self.min_size) & (width >= self.min_size)
        for k in np.unique(concept_of):
            ids = np.nonzero(concept_of == k)[0]
            r = ids[np.argmax(area[ids])]
            old = self.cells[k]["largest"]
            if old is None or area[r] > old.area:
                self.cells[k]["largest"] = self._cell(index, rgb, regions == r, k, int(area[r]), float(mean_d[r]),
                                                      cell_size)
            ok = ids[eligible[ids]]
            if len(ok):
                r = ok[np.argmin(mean_d[ok])]
                old = self.cells[k]["closest"]
                if old is None or mean_d[r] < old.distance:
                    self.cells[k]["closest"] = self._cell(index, rgb, regions == r, k, int(area[r]),
                                                          float(mean_d[r]), cell_size)

    def finish(self):
        """Concepts without an eligible closest cell use their largest one (with a warning)."""
        fallback = [k for k, c in enumerate(self.cells) if c["closest"] is None and c["largest"] is not None]
        for k in fallback:
            self.cells[k]["closest"] = self.cells[k]["largest"]
        if fallback and self.min_size is not None:
            warnings.warn(f"No cell reached min_size={self.min_size} px for concepts {fallback}; "
                          "their largest cell is used as the closest one.", stacklevel=3)
        return self.cells

    def _cell(self, index, rgb, mask, k, area, distance, cell_size):
        """A square view centred on the cell, with context, and the cell's mask in it."""
        H, W, _ = rgb.shape
        ch, cw = cell_size
        rows, cols = np.nonzero(mask)
        top, bottom = rows.min() * ch, (rows.max() + 1) * ch
        left, right = cols.min() * cw, (cols.max() + 1) * cw
        side = max(bottom - top, right - left) * (1 + 2 * self.context) + 8
        side = int(round(min(side, H, W)))                        # the view must fit in the image
        cy, cx = (top + bottom) / 2, (left + right) / 2
        y0 = int(round(np.clip(cy - side / 2, 0, H - side)))
        x0 = int(round(np.clip(cx - side / 2, 0, W - side)))
        y1, x1 = y0 + side, x0 + side
        crop = rgb[y0:y1, x0:x1]
        ys = np.minimum(((np.arange(y0, y1) + 0.5) / ch).astype(int), mask.shape[0] - 1)
        xs = np.minimum(((np.arange(x0, x1) + 0.5) / cw).astype(int), mask.shape[1] - 1)
        pixel_mask = mask[ys][:, xs]
        if side > CROP_SIDE:                                      # keep stored crops small
            crop = resize(np.ascontiguousarray(crop), CROP_SIDE, CROP_SIDE)
            pick = np.minimum((np.arange(CROP_SIDE) * side / CROP_SIDE).astype(int), side - 1)
            pixel_mask = pixel_mask[pick][:, pick]
        return Cell(index, k, area, distance, (int(round(top)), int(round(left)), int(round(bottom)),
                                               int(round(right))), np.ascontiguousarray(crop), pixel_mask)


def render_patch(cell, side=PATCH_SIDE, dim=0.3):
    """side x side x 3 uint8: the cell's crop, dimmed outside the cell, with a 1 px white line
    just outside its border. An empty (black) patch for None."""
    if cell is None:
        return np.zeros((side, side, 3), np.uint8)
    crop = resize(cell.crop, side, side).astype(np.float32)
    h, w = cell.mask.shape
    m = cell.mask[np.minimum((np.arange(side) * h / side).astype(int), h - 1)][
        :, np.minimum((np.arange(side) * w / side).astype(int), w - 1)]
    out = crop * np.where(m, 1.0, dim)[..., None]
    grown = m.copy()
    grown[1:] |= m[:-1]; grown[:-1] |= m[1:]; grown[:, 1:] |= m[:, :-1]; grown[:, :-1] |= m[:, 1:]
    out[grown & ~m] = 255
    return np.clip(np.round(out), 0, 255).astype(np.uint8)
