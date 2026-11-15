# SPDX-License-Identifier: MIT
"""Turn explanations, label maps and heatmaps into images (numpy arrays, uint8).

>>> exp = nave.explain(image)
>>> picture = nave.draw.concepts(exp)                      # the explanation map
>>> borders = nave.draw.borders(exp.superpixels, color="black", width=2)    # RGBA layer
"""
import colorsys

import numpy as np

from ._images import as_rgb
from .interp import resample

# matplotlib's tab10, then tab20's light tones, like the paper's figures
_TAB20 = ["1f77b4", "ff7f0e", "2ca02c", "d62728", "9467bd", "8c564b", "e377c2", "7f7f7f", "bcbd22", "17becf",
          "aec7e8", "ffbb78", "98df8a", "ff9896", "c5b0d5", "c49c94", "f7b6d2", "c7c7c7", "dbdb8d", "9edae5"]


def _rgba(color):
    from matplotlib.colors import to_rgba
    return np.round(np.array(to_rgba(color)) * 255).astype(np.uint8)


ORDERED_COLORMAP = "turbo"


def make_palette(n, name="tab20"):
    """`n` colours (uint8, (n, 3)). "tab20" (the paper's style): distinct colours, generated
    hues beyond 20. "ordered": evenly spaced along a continuous colormap, for reordered concepts.
    Any matplotlib colormap name or a list of colours also works."""
    if isinstance(name, np.ndarray):                       # explicit colours
        base = [tuple(int(v) for v in c[:3]) for c in name]
        name = None
    if isinstance(name, str) and name == "ordered":
        name = ORDERED_COLORMAP
    if name is None:
        pass
    elif isinstance(name, (list, tuple)):
        base = [tuple(_rgba(c)[:3]) for c in name]
    elif name == "tab20":
        base = [tuple(int(h[i:i + 2], 16) for i in (0, 2, 4)) for h in _TAB20]
    else:
        from matplotlib import colormaps
        cmap = colormaps[name]
        count = getattr(cmap, "N", 256)
        if count > 64:                                     # continuous: n evenly spaced colours
            lo, hi = (0.05, 0.95) if name == ORDERED_COLORMAP else (0.0, 1.0)
            base = [tuple(int(round(255 * c)) for c in cmap(lo + (hi - lo) * i / max(n - 1, 1))[:3])
                    for i in range(n)]
        else:
            base = [tuple(int(round(255 * c)) for c in cmap(i)[:3]) for i in range(min(count, n))]
    cols, i = list(base), 0
    while len(cols) < n:                                   # golden-ratio hues
        hue = (i * 0.618033988749895) % 1.0
        sat, val = (0.65, 0.9) if i % 2 == 0 else (0.45, 0.7)
        cols.append(tuple(int(round(255 * c)) for c in colorsys.hsv_to_rgb(hue, sat, val)))
        i += 1
    return np.array(cols[:n], np.uint8)


def borders(labels, color="black", width=1):
    """RGBA layer (H, W, 4) with the boundaries between labels, `width` px wide."""
    labels = np.asarray(labels)
    H, W = labels.shape
    edge = np.zeros((H, W), bool)
    edge[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    edge[:-1, :] |= labels[:-1, :] != labels[1:, :]
    for i in range(int(width) - 1):                        # grow alternately down-right and up-left
        g = edge.copy()
        if i % 2 == 0:
            g[:, 1:] |= edge[:, :-1]; g[1:, :] |= edge[:-1, :]; g[1:, 1:] |= edge[:-1, :-1]
        else:
            g[:, :-1] |= edge[:, 1:]; g[:-1, :] |= edge[1:, :]; g[:-1, :-1] |= edge[1:, 1:]
        edge = g
    out = np.zeros((H, W, 4), np.uint8)
    out[edge] = _rgba(color)
    return out


def average_color(image, labels):
    """(H, W, 3): every segment filled with its mean colour (colored superpixels)."""
    rgb = as_rgb(image)
    flat = np.asarray(labels).ravel()
    n = int(flat.max()) + 1
    counts = np.maximum(np.bincount(flat, minlength=n), 1)
    means = np.stack([np.bincount(flat, weights=rgb[..., c].ravel().astype(np.float64), minlength=n)
                      for c in range(3)], 1) / counts[:, None]
    return np.round(means[np.asarray(labels)]).astype(np.uint8)


def concepts(explanation, uncertainty=None, palette=None, background=None):
    """The NAVE explanation map: every concept in its own colour (the paper's style).

    On request, the uncertainty is drawn as opacity: alpha = certainty = 1 - uncertainty, so
    confident concepts are opaque and uncertain pixels fade out.

    Parameters
    ----------
    explanation : an `ImageExplanation`, or an (H, W) concept map
    uncertainty : None (default: plain colours), "opacity" (use the explanation's uncertainty
        map), or an (H, W) array in [0, 1]
    palette : None (default: the explanation's own palette if it has one, "ordered" for
        reordered explanations, else "tab20"), a (K, 3) uint8 array, "tab20",
        "ordered" (evenly spaced colours along a continuous colormap, so neighbouring concepts
        get neighbouring colours), a matplotlib colormap name, or a list of colours
    background : None, an image or a colour to show through uncertain pixels

    Returns
    -------
    (H, W, 3) uint8 without opacity or with a background; (H, W, 4) RGBA otherwise
    """
    concept_map = getattr(explanation, "concepts", explanation)
    if concept_map is None:
        raise ValueError("This explanation has no image-wide concepts (single pass over windows).")
    concept_map = np.asarray(concept_map)
    if isinstance(uncertainty, str):
        if uncertainty != "opacity":
            raise ValueError("uncertainty must be None, 'opacity' or an array.")
        uncertainty = getattr(explanation, "uncertainty", None)
        if uncertainty is None:
            raise ValueError("This explanation has no uncertainty map: explain(..., return_uncertainty=True).")
    if palette is None:
        palette = getattr(explanation, "palette", None)
    if palette is None:
        palette = "ordered" if getattr(explanation, "order", None) is not None else "tab20"
    K = len(explanation.prototypes) if getattr(explanation, "prototypes", None) is not None \
        else int(concept_map.max()) + 1
    rgb = make_palette(max(K, int(concept_map.max()) + 1), palette)[concept_map]
    if uncertainty is None:
        return rgb
    alpha = np.round(255 * (1.0 - np.clip(np.asarray(uncertainty, np.float32), 0, 1))).astype(np.uint8)
    layer = np.concatenate([rgb, alpha[..., None]], axis=2)
    if background is None:
        return layer
    if isinstance(background, str) or np.ndim(background) == 1:
        background = np.broadcast_to(_rgba(background)[:3], rgb.shape)
    return overlay(background, layer, alpha=1.0)


def heatmap(values, cmap=None, vmin=None, vmax=None, center=None, size=None):
    """(H, W, 3) colour image of a 2-D map.

    Defaults follow the content: maps with negative values (saliency) use the diverging
    "coolwarm" centred on 0; maps in [0, 1] (entropy) use "inferno".
    size : optional (height, width) to resize to (nearest neighbour, to keep cells visible)
    """
    from matplotlib import colormaps
    v = np.asarray(values, np.float32)
    if size is not None and v.shape != tuple(size):
        v = resample(v, size[0], size[1], "nearest")
    finite = v[np.isfinite(v)]
    lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
    if center is None and lo < 0 and vmin is None and vmax is None:
        center = 0.0
    if cmap is None:
        cmap = "coolwarm" if center is not None else "inferno"
    if center is not None:
        half = max(abs((vmax if vmax is not None else hi) - center), abs(center - (vmin if vmin is not None else lo)))
        vmin, vmax = center - half, center + half
    vmin = lo if vmin is None else vmin
    vmax = hi if vmax is None else vmax
    t = np.clip((v - vmin) / max(vmax - vmin, 1e-12), 0.0, 1.0)
    return np.round(colormaps[cmap](t)[..., :3] * 255).astype(np.uint8)


def overlay(image, layer, alpha=0.5):
    """Blend a layer onto an image. RGBA layers use their own alpha times `alpha`
    (alpha=1 by default makes sense for borders); RGB layers are blended with `alpha`."""
    base = as_rgb(image).astype(np.float32)
    layer = np.asarray(layer)
    if layer.shape[:2] != base.shape[:2]:
        raise ValueError(f"Layer size {layer.shape[:2]} differs from image size {base.shape[:2]}.")
    if layer.shape[2] == 4:
        a = layer[..., 3:4].astype(np.float32) / 255.0 * (1.0 if alpha is None else alpha)
        top = layer[..., :3].astype(np.float32)
    else:
        a, top = float(alpha), layer.astype(np.float32)
    return np.round(base * (1 - a) + top * a).astype(np.uint8)


def colorbar(cmap="coolwarm", vmin=-1.0, vmax=1.0, size=(16, 256), vertical=False):
    """A colour strip (h, w, 3) from vmin (left/bottom) to vmax, to save next to a heatmap."""
    h, w = size
    ramp = np.linspace(vmin, vmax, h if vertical else w)
    values = (np.repeat(ramp[::-1, None], w, 1) if vertical else np.repeat(ramp[None], h, 0))
    return heatmap(values, cmap, vmin=vmin, vmax=vmax)
