# SPDX-License-Identifier: MIT
"""`nave_xai.saliency`: concept-based class saliency, propagated through NAVE explanations.

1. At every position of the deepest selected block (before any pooling), take the cosine
   between its feature vector and the classifier's weight vector of the target class.
2. Explain that block alone with NAVE and replace each value by its region's mean.
3. Go one block shallower: upsample the map to that block's grid, explain with the
   cumulative block set (for N234: N4, then N34, then N234) and average per region again.
4. Upsample the last map to the image.

The map is piecewise constant over regions the encoder treats as coherent.
"""
from dataclasses import dataclass, field

import numpy as np

from ._engine import AUTO_MAX_PIXELS, Tiling
from ._images import normalize as normalize_image
from .clustering import BayesianKMeans, KMeans, PCA, as_clusterer, fresh_copy
from .components import connected_components, segment_means
from .composite import CompositeFeatures
from .interp import resample


@dataclass
class SaliencyResult:
    saliency: np.ndarray                 # (H, W) float32 in [-1, 1]
    raw: np.ndarray                      # cosine on the deepest block's grid, before averaging
    levels: list = field(default_factory=list)          # map after each level, on its own grid
    regions: list = field(default_factory=list)         # region map used at each level (same grids)
    level_blocks: list = field(default_factory=list)    # blocks used at each level
    target: int = None
    score: float = None                  # classifier score of the target class


def as_head(classifier):
    """Classifier weights from a torch.nn.Linear, a (classes, C) array, a (weight, bias) pair or a dict."""
    if classifier is None:
        return None
    if isinstance(classifier, dict):
        return {"weight": np.asarray(classifier["weight"], np.float32),
                "bias": None if classifier.get("bias") is None else np.asarray(classifier["bias"], np.float32)}
    if isinstance(classifier, (tuple, list)) and len(classifier) == 2:
        return {"weight": np.asarray(classifier[0], np.float32), "bias": np.asarray(classifier[1], np.float32)}
    if hasattr(classifier, "weight"):                        # torch.nn.Linear
        bias = getattr(classifier, "bias", None)
        to_np = lambda t: t.detach().float().cpu().numpy()
        return {"weight": to_np(classifier.weight), "bias": None if bias is None else to_np(bias)}
    w = np.asarray(classifier, np.float32)
    if w.ndim != 2:
        raise ValueError("classifier weights must be a (classes, channels) matrix.")
    return {"weight": w, "bias": None}


def _level_clusterer(spec, k, seed, alpha, iterations, burn_in):
    algo = as_clusterer(spec, k, seed, alpha, iterations, burn_in)
    if isinstance(algo, (KMeans, BayesianKMeans, PCA)) and isinstance(spec, (str, type(None))):
        return algo                                                        # a fresh built-in one
    return fresh_copy(algo, seed)


def saliency(image, backbone, names, k=10, *, classifier=None, target=None, upsample="cubic",
             normalize=True, scale=True, clusterer=None, alpha=1.0, iterations=200, burn_in=100,
             sample_size=20000, connectivity=8, seed=0):
    """Concept-based class saliency of one image (used by `NAVEImage.saliency`).

    image : (H, W, 3) uint8 (normalised here with the backbone's mean and std) or an already
        transformed float image; at most 1024 x 1024 pixels in total
    backbone, names : the resolved backbone and its layers, shallowest first
    classifier : None (the model's own head) or as accepted by `as_head`
    target : class index; None = the class the model predicts

    Returns a SaliencyResult.
    """
    H, W = image.shape[:2]
    if H * W > AUTO_MAX_PIXELS:
        raise ValueError(f"saliency works on images of up to 1024 x 1024 pixels in total (got {H} x {W}); "
                         "resize the image first (e.g. with transforms=).")
    tiling = Tiling.plan(H, W, None, backbone)
    if tiling.n != 1:
        raise ValueError("This model's fixed input size is smaller than the image; resize the image first.")
    padded = tiling.padded(image)
    x = normalize_image(padded, backbone.mean, backbone.std) if padded.dtype == np.uint8 \
        else np.ascontiguousarray(padded.transpose(2, 0, 1)[None], np.float32)
    outputs = backbone(x, list(names) + [n for n in backbone.extra_outputs if n not in names])

    head = as_head(classifier) or backbone.head()
    if head is None:
        raise ValueError("This model has no classifier head (DINO and random weights have none); "
                         "pass classifier= (a torch.nn.Linear or a (classes, channels) weight matrix).")
    deep = backbone.head_features(outputs[names[-1]][0])                   # (C, h, w)
    C, gh, gw = deep.shape
    if head["weight"].shape[1] != C:
        raise ValueError(f"The classifier expects {head['weight'].shape[1]} channels, "
                         f"the deepest block {names[-1]!r} has {C}.")
    valid = (int(np.ceil(gh * H / tiling.tile_h)), int(np.ceil(gw * W / tiling.tile_w)))
    if classifier is None:
        logits = backbone.logits(outputs, valid)
    else:
        pooled = deep[:, :valid[0], :valid[1]].reshape(C, -1).mean(1)
        logits = head["weight"] @ pooled + (0 if head["bias"] is None else head["bias"])
    if target is None:
        if logits is None:
            raise ValueError("Cannot tell which class the model predicts; pass target=.")
        target = int(np.argmax(logits))
    w = head["weight"][target]
    raw = np.tensordot(w, deep, axes=(0, 0)) / (np.linalg.norm(deep, axis=0) * np.linalg.norm(w) + 1e-12)
    raw = raw.astype(np.float32)[:valid[0], :valid[1]]

    current = np.tensordot(w, deep, axes=(0, 0)) / (np.linalg.norm(deep, axis=0) * np.linalg.norm(w) + 1e-12)
    current = current.astype(np.float32)                     # full grid, padding included
    levels, regions, level_blocks = [], [], []
    for step, j in enumerate(range(len(names) - 1, -1, -1)):
        subset = names[j:]
        comp = CompositeFeatures([outputs[n][0] for n in subset], upsample, normalize, scale)
        if current.shape != (comp.height, comp.width):
            current = resample(current, comp.height, comp.width, upsample)
        algo = _level_clusterer(clusterer, k, seed + step, alpha, iterations, burn_in)
        rng = np.random.default_rng([seed, step])
        algo.fit(comp.sample(sample_size, rng))
        labels = comp.scores(algo).argmax(2)
        seg, n = connected_components(labels, connectivity)
        current = segment_means(current, seg, n)
        rows = int(np.ceil(comp.height * H / tiling.tile_h))      # cells covering the image, not padding
        cols = int(np.ceil(comp.width * W / tiling.tile_w))
        levels.append(current[:rows, :cols])
        regions.append(seg[:rows, :cols])
        level_blocks.append(list(subset))

    final = resample(current, tiling.tile_h, tiling.tile_w, upsample)[:H, :W]
    return SaliencyResult(np.clip(final, -1.0, 1.0).astype(np.float32), raw, levels, regions,
                          level_blocks, target, None if logits is None else float(logits[target]))
