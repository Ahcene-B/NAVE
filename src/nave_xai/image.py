# SPDX-License-Identifier: MIT
"""`NAVEImage`: concepts extracted from a vision encoder's activations, and the explanations they give.

    >>> from nave_xai import NAVEImage
    >>> explainer = NAVEImage()                       # ViT-Ti/16, ImageNet weights shipped with the package
    >>> exp = explainer.explain(image)           # fits on this image, then explains it
    >>> exp.explanation                          # (H, W) concept of every pixel

NAVE (Boubekki, Fadel, Mair, ECML PKDD 2026) combines the feature maps of several layers into
composite pixels (Algorithm 1) and projects them onto K concepts (k-means by default).
"""
import math
import warnings
from dataclasses import dataclass

import numpy as np

from . import memory
from ._engine import Tiling, _Engine
from ._images import as_rgb, normalize as normalize_image, resize
from ._progress import make_progress, span
from ._shared import _ConceptsBase, _Subset, _identity
from ._summary import KINDS, CellTracker, classes_by_frequency, correlation as correlate, render_patch
from .clustering import BayesianKMeans, ForeignClusterer, KMeans, PCA, is_fitted
from .components import connected_components
from .interp import check_mode, resample
from .models import get_backbone


# --------------------------------------------------------------------------
# ImageExplanation
# --------------------------------------------------------------------------
@dataclass
class ImageExplanation:
    """The NAVE explanation of one image.

    Attributes
    ----------
    explanation : (H, W) int32, the concept of every pixel (the explanation map)
    uncertainty : (H, W) float32 in [0, 1], normalised entropy of the concept assignment
        (with uncertainty=True), else None
    K : number of concepts, e.g. for ``np.bincount(exp.explanation.ravel(), minlength=exp.K)``
    """

    explanation: np.ndarray
    uncertainty: np.ndarray = None
    K: int = 0

    def segmentation(self):
        """(H, W) int32 connected regions of the explanation map, like skimage.segmentation.slic."""
        return connected_components(self.explanation)[0]

    def border(self, width=1):
        """(H, W) int32 mask (0 and 1) of the inner margin, `width` px wide, of every region."""
        seg = self.segmentation()
        out = np.zeros(seg.shape, np.int32)
        for d in range(1, int(width) + 1):
            diff = seg[d:] != seg[:-d]
            out[d:][diff] = 1
            out[:-d][diff] = 1
            diff = seg[:, d:] != seg[:, :-d]
            out[:, d:][diff] = 1
            out[:, :-d][diff] = 1
        return out


# --------------------------------------------------------------------------
# NAVEImage
# --------------------------------------------------------------------------
class NAVEImage(_ConceptsBase):
    """NAVE explainer: concepts from a vision encoder, and explanations of images with them.

    Parameters
    ----------
    model : "vitt16" (ViT-Ti/16, shipped), "resnet18", "resnet50", "vits16", a torch.nn.Module,
        a Backbone, a path to an .onnx file, or a function returning feature maps
    training : weights of a built-in model: "random", "imagenet" or "dino" (vitt16: random, imagenet)
    layers : built-in models: numbers (``[3, 4]``) or paper notation (``"N34"``, ``"N:9-12"``);
        None: N34 for ResNets, the last block for ViTs. Other models: module names.
    k : number of concepts per projector (an upper bound with projector="bayesian")
    projector : None/"kmeans", "bayesian" (Dirichlet-prior k-means, Gibbs sampling), "pca", or
        a scikit-learn-style object. Clusterers assign a pixel to the closest prototype,
        projections (PCA, NMF, ...) to the component with the largest absolute value.
    bayes_alpha, bayes_iter, bayes_burn_in : Dirichlet concentration, Gibbs sweeps and
        discarded sweeps of projector="bayesian"
    upsample : "nearest" (the paper's blocky maps), "linear" or "cubic": interpolation of deeper
        layers and of concept scores to pixels
    window : "auto" (windows when the longer side is at least `window_factor` times the model's
        input size, 224 px for the built-in models: windows twice the input size, every half
        window, each pixel from the window whose centre is nearest), None (always whole), or a
        window size in px
    window_factor : see `window`
    features : built-in ViTs: "tokens" (the blocks' outputs, default) or "keys" (the attention keys
        of the same blocks, as in LOST and TokenCut). Saliency and class predictions always use
        the blocks' outputs, which the classifier reads.
    fit_on : "all" (every composite pixel: exact if they fit in memory, else streamed through
        mini-batch k-means, with a warning) or "sample" (`sample_size` composite pixels)
    sample_size : composite pixels used with fit_on="sample" (and to seed streaming)
    epochs : streaming passes when fit_on="all" falls back to mini-batch k-means
    normalize : per-pixel L2 normalisation of each layer (Algorithm 1, line 9)
    scale : C_min / (1 + C_b) scaling of each layer (Algorithm 1, line 10)
    uncertainty_temp : softmax temperature of the uncertainty (default 0.1). "auto": 1 for
        probabilistic projectors, else the median gap between the two best concept scores
    transforms : "default" (ImageNet-style normalisation of the model) or a function, or a list
        of functions applied in order, from image to image (e.g. a resizing lambda). They get an
        (H, W, 3) uint8 array. If the last one returns uint8 (or floats in [0, 1]), the model's
        default normalisation follows; otherwise its output is used as the normalised input.
        Use only resizing and colour operations: maps are returned at the original size.
    classifier : for the classes, correlations and saliency: None (the model's own head), a
        torch.nn.Linear, a (classes, C) weight matrix, a (weight, bias) pair, or a function
        image -> class scores or class id
    class_names : names of the classes for `summarize` (a list or dict)
    seed : random seed
    device : for PyTorch models
    progress : False, True (progress bar) or a callable(fraction, text)

    Attributes (after fitting)
    --------------------------------
    fitted : True after an explicit `fit`
    prototypes : (K, D) concept prototypes (PCA: components)
    palette : (K, 3) uint8 concept colours (Spectral by default, along the concept order)
    patches : (K, 2, 32, 32, 3) uint8, the closest and the largest cell of every concept in the
        fitting set: dimmed outside the cell, with a 1 px white line around it
    classes : (C,) the groups given to `fit`, else the model's predicted classes
    correlation : (K, C) Pearson correlation between concept proportions and `classes`
    """

    _PER_CONCEPT = ("patches", "_cells")

    def __init__(self, model="vitt16", training="imagenet", layers=None, k=10, projector=None,
                 bayes_alpha=1.0, bayes_iter=200, bayes_burn_in=100, upsample="cubic", window="auto",
                 fit_on="all", sample_size=20000, epochs=1, normalize=True, scale=True,
                 uncertainty_temp=0.1, transforms="default", classifier=None, class_names=None,
                 window_factor=3, features="tokens", seed=0, device="cpu", progress=False):
        check_mode(upsample)
        if fit_on not in ("all", "sample"):
            raise ValueError("fit_on must be 'all' or 'sample'.")
        self.model, self.training, self.layers, self.k, self.projector = model, training, layers, k, projector
        self.bayes_alpha, self.bayes_iter, self.bayes_burn_in = bayes_alpha, bayes_iter, bayes_burn_in
        self.upsample, self.window, self.fit_on = upsample, window, fit_on
        self.sample_size, self.epochs, self.normalize, self.scale = sample_size, epochs, normalize, scale
        self.uncertainty_temp, self.transforms = uncertainty_temp, transforms
        self.classifier, self.class_names = classifier, class_names
        self.window_factor, self.features = window_factor, features
        self.seed, self.device, self.progress = seed, device, progress
        self.fitted = False
        self.prototypes = self.palette = self.patches = self.classes = self.correlation = None
        self._backbone = None
        self._stack = None

    # -- model, transforms, engine -----------------------------------------------
    @property
    def backbone(self):
        if self._backbone is None:
            self._backbone = get_backbone(self.model, self.training, self.seed, self.device)
            self.layers_ = self._backbone.resolve_blocks(self.layers)
            self._token_layers = list(self.layers_)               # what the classifier reads (saliency, classes)
            if self.features == "keys":
                from .models.backbone import BuiltinBackbone
                bb = self._backbone
                if not (isinstance(bb, BuiltinBackbone) and bb.name.startswith("vit")):
                    raise ValueError('features="keys" is available for the built-in ViTs ("vitt16", "vits16"); '
                                     'for your own models, name the layer to read (e.g. a key projection).')
                self.layers_ = ["k" + n[1:] for n in self.layers_]  # the attention keys of the same blocks
            elif self.features != "tokens":
                raise ValueError('features must be "tokens" or "keys".')
        return self._backbone

    def _transform(self, rgb):
        """The network's view of an image: uint8 (normalised later) or a normalised float image."""
        if isinstance(self.transforms, str):
            if self.transforms != "default":
                raise ValueError("transforms must be 'default', a function or a list of functions.")
            return rgb
        x = rgb
        for step in (self.transforms if isinstance(self.transforms, (list, tuple)) else [self.transforms]):
            x = step(x)
        if hasattr(x, "detach"):
            x = x.detach().cpu().numpy()
        x = np.asarray(x)
        if x.ndim == 3 and x.shape[0] in (1, 3) and x.shape[2] not in (1, 3):
            x = x.transpose(1, 2, 0)
        if x.ndim == 2 or x.shape[2] == 1:
            x = np.repeat(x.reshape(x.shape[0], x.shape[1], 1), 3, axis=2)
        if x.dtype == np.uint8:
            return np.ascontiguousarray(x)
        x = x.astype(np.float32)
        if x.min() >= 0.0 and x.max() <= 1.0:              # [0, 1] floats: the default normalisation
            mean, std = np.asarray(self.backbone.mean, np.float32), np.asarray(self.backbone.std, np.float32)
            x = (x - mean) / std
        return np.ascontiguousarray(x)

    def _prepare(self, image):
        rgb = as_rgb(image)
        return rgb, self._transform(rgb)

    def _tiling(self, H, W, window=None):
        return Tiling.plan(H, W, self.window if window is None else window, self.backbone, self.window_factor)

    def _engine(self, net, uncertainty=False):
        tiling = self._tiling(net.shape[0], net.shape[1])
        opts = dict(upsample=self.upsample, normalize=self.normalize, scale=self.scale, block_weights=None,
                    k=self.k, seed=self.seed, sample_size=self.sample_size, fit_on=self.fit_on,
                    epochs=self.epochs, two_pass=True, connectivity=8, entropy=uncertainty,
                    temperature=1.0)
        return _Engine(net, self.backbone, self.layers_, tiling, opts, _identity)

    def _check_memory(self, shapes, uncertainty=False):
        bb = self.backbone
        if bb.name not in memory.MODELS or not shapes:
            return
        H, W = max(shapes, key=lambda s: s[0] * s[1])[:2]
        tiling = self._tiling(H, W)
        ids = [bb.blocks.index(n) + 1 if n in bb.blocks else 1 for n in self._token_layers]
        need = memory.estimate(bb.name, ids, tiling.tile_h, tiling.tile_w, tiling.n, H * W, self.k, True,
                               self.fit_on == "all", uncertainty, self.sample_size)
        memory.warn_if_too_large(need, memory.advice(bb.name, ids, None if tiling.n == 1 else tiling.tile_h))

    @staticmethod
    def _is_single(images):
        return (isinstance(images, np.ndarray) and images.ndim in (2, 3)) or \
            (hasattr(images, "mode") and hasattr(images, "size"))

    @classmethod
    def _as_list(cls, images):
        if isinstance(images, np.ndarray) and images.ndim == 4:
            return list(images)
        if cls._is_single(images):
            return [images]
        return list(images)

    # -- fitting ---------------------------------------------------------------
    def fit(self, images, groups=None, merge_groups=True, reorder="ward"):
        """Extract concepts from `images`.

        images : an (N, H, W, 3) or (N, H, W) array (uint8, or float in [0, 1]), or a list of images
        groups : one label per image: one projector is fitted per group, and the groups become
            the `classes` of the correlation
        merge_groups : merge the groups' projectors into one (`predict` then ignores groups)
        reorder : "ward", "average", "single" or None: order of the concept ids. With groups,
            concepts are ordered by group first, then within each group, and every group gets a
            colour family.
        """
        self._fit(images, groups, merge_groups, reorder)
        self.fitted = True
        return self

    def _fit(self, images, groups, merge_groups, reorder):
        images = self._as_list(images)
        report = make_progress(self.progress)
        self.backbone
        self._check_memory([as_rgb(im).shape for im in images[:64]])
        if groups is not None:
            groups = np.asarray(groups)
            if len(groups) != len(images):
                raise ValueError("groups must have one label per image.")
            labels = list(dict.fromkeys(groups.tolist()))
        else:
            labels = [None]
        projectors, sizes = [], []
        for i, g in enumerate(labels):
            members = images if g is None else [im for im, gi in zip(images, groups) if gi == g]
            proj = self._new_projector(i)
            part = span(report, 0.8 * i / len(labels), 0.8 * (i + 1) / len(labels))
            probe = self._fit_projector(proj, members, part)
            sizes.append(proj.scores(probe[:1]).shape[1])
            projectors.append(proj)
        self._set_projectors(projectors, sizes, labels, groups, merge_groups)
        self._scan(images, groups, span(report, 0.8, 0.97))
        self._finish_fit(groups, reorder)
        report(1.0, "fitted")

    def _kept_chunks(self, engine):
        """Yield the composite pixels of every window's kept region, chunk by chunk."""
        for tile in engine.tiling.tiles():
            comp = engine.features(tile)
            rows, cols = engine.grid_range(comp, tile)
            yield from comp.region(rows, cols)

    def _sample(self, images, n, report):
        nets = [self._prepare(im)[1] for im in images]
        area = sum(x.shape[0] * x.shape[1] for x in nets)
        parts = []
        for i, net in enumerate(nets):
            share = max(1, int(round(n * net.shape[0] * net.shape[1] / area)))
            parts.append(self._engine(net).sample(share, _identity))
            report((i + 1) / len(nets), f"sampling {i + 1}/{len(nets)}")
        return np.concatenate(parts)

    def _count_cells(self, images):
        """Number and size of all kept composite pixels, from the first window's grid."""
        first = self._engine(self._prepare(images[0])[1])
        tile = next(first.tiling.tiles())
        comp = first.features(tile)
        stride, dim = first.tiling.tile_h / comp.height, comp.dim
        total = 0
        for im in images:
            net = self._prepare(im)[1]
            tiling = self._tiling(net.shape[0], net.shape[1])
            for t in tiling.tiles():
                total += (math.ceil(t.rows[1] / stride) - int(t.rows[0] // stride)) * \
                         (math.ceil(t.cols[1] / stride) - int(t.cols[0] // stride))
        return total, dim

    def _fit_projector(self, proj, images, report):
        """Fit one projector on images; return a few composite pixels (to probe its size)."""
        if is_fitted(proj):
            return self._sample(images[:1], 16, _identity)
        if self.fit_on == "all" and isinstance(proj, BayesianKMeans):
            warnings.warn(f"projector='bayesian' with fit_on='all': Gibbs sampling on every composite pixel "
                          f"would take very long; fitted on {self.sample_size} sampled composite pixels.",
                          stacklevel=4)
        elif self.fit_on == "all":
            total, dim = self._count_cells(images)
            need, free = 4 * total * dim, memory.available()
            if free is None or need <= 0.5 * free:
                X = np.empty((total, dim), np.float32)
                at = 0
                for i, im in enumerate(images):
                    for chunk in self._kept_chunks(self._engine(self._prepare(im)[1])):
                        X[at:at + len(chunk)] = chunk
                        at += len(chunk)
                    report(0.5 * (i + 1) / len(images), f"composite pixels {i + 1}/{len(images)}")
                report(0.5, "projecting")
                proj.fit(X[:at])
                return X[:16].copy()
            gb = memory.GB
            if isinstance(proj, PCA):                       # covariance accumulates exactly
                self._stream(proj, images, 1, report)
                return self._sample(images[:1], 16, _identity)
            if isinstance(proj, KMeans) or (isinstance(proj, ForeignClusterer) and proj.supports_partial_fit):
                warnings.warn(f"fit_on='all': the composite pixels need {need / gb:.1f} GB and "
                              f"{free / gb:.1f} GB is free; streamed them through mini-batch k-means "
                              f"({self.epochs} epoch(s)) instead of exact k-means.", stacklevel=4)
                if isinstance(proj, KMeans):
                    proj.fit(self._sample(images, self.sample_size, span(report, 0.0, 0.3)))
                self._stream(proj, images, self.epochs, span(report, 0.3, 1.0))
                return self._sample(images[:1], 16, _identity)
            warnings.warn(f"fit_on='all': the composite pixels need {need / gb:.1f} GB and {free / gb:.1f} GB "
                          f"is free, and this projector cannot be streamed; fitted on {self.sample_size} "
                          "sampled composite pixels.", stacklevel=4)
        X = self._sample(images, self.sample_size, span(report, 0.0, 0.7))
        proj.fit(X)
        report(1.0, "projecting")
        return X[:16]

    def _stream(self, proj, images, epochs, report):
        rng = np.random.default_rng(self.seed)
        for e in range(epochs):
            for i, im in enumerate(images):
                for chunk in self._kept_chunks(self._engine(self._prepare(im)[1])):
                    proj.partial_fit(chunk[rng.permutation(len(chunk))])
                report((e * len(images) + i + 1) / (epochs * len(images)), f"streaming {i + 1}/{len(images)}")

    # -- one pass with the fitted concepts ----------------------------------------
    def _grid_map(self, engine, scorer):
        """Concept map, distance to the own prototype, and scores' best gaps, on the image's grid."""
        tiling = engine.tiling
        maps, dists, gaps, stride = None, None, [], None
        for tile in tiling.tiles():
            comp = engine.features(tile)
            if stride is None:
                stride = tiling.tile_h / comp.height
                gh, gw = math.ceil(tiling.H / stride), math.ceil(tiling.W / stride)
                maps, dists = np.zeros((gh, gw), np.int32), np.zeros((gh, gw), np.float32)
            (r0, r1), (c0, c1) = engine.grid_range(comp, tile)
            y0, x0 = int(round(tile.out_rows[0] / stride)), int(round(tile.out_cols[0] / stride))
            labels, dist = [], []
            for X in comp.region((r0, r1), (c0, c1)):
                s = scorer.scores(X)
                lab = s.argmax(1)
                labels.append(lab)
                if self.prototypes is not None and self._stack.centroid_based:
                    dist.append(np.sqrt(((X - self.prototypes[lab]) ** 2).sum(1)))
                else:
                    dist.append(-s.max(1))
                if len(gaps) < 4 and s.shape[1] > 1:
                    top2 = np.partition(s, -2, axis=1)[:, -2:]
                    gaps.append(float(np.median(top2[:, 1] - top2[:, 0])))
            h, w = r1 - r0, c1 - c0
            hh, ww = min(h, gh - y0), min(w, gw - x0)
            maps[y0:y0 + hh, x0:x0 + ww] = np.concatenate(labels).reshape(h, w)[:hh, :ww]
            dists[y0:y0 + hh, x0:x0 + ww] = np.concatenate(dist).reshape(h, w)[:hh, :ww]
        return maps, dists, gaps, stride

    def _scan(self, images, groups, report, min_size=None, patches_only=False):
        """Concept sizes, temperature, patches, classes and correlation, in one pass."""
        K = sum(self._stack.sizes)
        tracker = CellTracker(K, min_size)
        sizes, gaps, proportions, predictions = np.zeros(K), [], [], []
        predict = groups is None and not patches_only and self._can_predict()
        for i, im in enumerate(images):
            rgb, net = self._prepare(im)
            maps, dists, g, stride = self._grid_map(self._engine(net), self._stack)
            gaps += g
            counts = np.bincount(maps.ravel(), minlength=K)
            sizes += counts
            proportions.append(counts / max(counts.sum(), 1))
            cell = (stride * rgb.shape[0] / net.shape[0], stride * rgb.shape[1] / net.shape[1])
            tracker.update(i, rgb, maps, dists, cell)
            if predict:
                predictions.append(self._predict_class(rgb, net))
            report((i + 1) / len(images), f"patches and classes {i + 1}/{len(images)}")
        cells = tracker.finish()
        self._cells = cells
        self.patches = np.stack([np.stack([render_patch(c[kind]) for kind in KINDS]) for c in cells])
        if patches_only:
            return
        self._sizes = sizes
        if self.uncertainty_temp != "auto":
            self.temperature_ = float(self.uncertainty_temp)
        else:
            self.temperature_ = 1.0 if self._stack.probabilistic or not gaps else max(float(np.median(gaps)), 1e-6)
        self._proportions = np.array(proportions)
        labels = groups if groups is not None else (
            np.array(predictions) if predict and all(p is not None for p in predictions) else None)
        if predict and labels is None:
            warnings.warn("The model gave no class prediction for some images: no classes and correlation. "
                          "Pass classifier= to NAVE.", stacklevel=4)
        if labels is None or len(images) < 2 or len(set(np.asarray(labels).tolist())) < 2:
            self.classes = None if labels is None else classes_by_frequency(labels)
            self.correlation = None                                # needs several images and classes
        else:
            self.classes = classes_by_frequency(labels)
            self.correlation = correlate(self._proportions, labels, self.classes)

    # -- predicted classes -----------------------------------------------------------
    def _can_predict(self):
        bb = self.backbone
        return self.classifier is not None or bb.head() is not None or type(bb).__name__ == "TorchBackbone"

    def _predict_class(self, rgb, net):
        """The class the model predicts for an image. For large images: the windows' logits,
        averaged with weights equal to their share of real (not padded) pixels."""
        clf = self.classifier
        if clf is not None and callable(clf) and not hasattr(clf, "weight") \
                and not isinstance(clf, (np.ndarray, list, tuple, dict)):
            out = np.asarray(clf(rgb))
            return int(out) if out.ndim == 0 else int(np.argmax(out))
        tiling = self._tiling(net.shape[0], net.shape[1])
        padded = tiling.padded(net)
        total, weights = None, 0.0
        for t in tiling.tiles():
            y0, x0 = t.out_rows[0] - t.rows[0], t.out_cols[0] - t.cols[0]       # window origin in the image
            window = padded[y0 + tiling.margin:y0 + tiling.margin + tiling.tile_h,
                            x0 + tiling.margin:x0 + tiling.margin + tiling.tile_w]
            real = (max(0, min(tiling.H, y0 + tiling.tile_h) - max(0, y0)) *
                    max(0, min(tiling.W, x0 + tiling.tile_w) - max(0, x0))) / (tiling.tile_h * tiling.tile_w)
            logits = self._window_logits(window, real)
            if logits is None:
                return None
            total = logits * real if total is None else total + logits * real
            weights += real
        return int(np.argmax(total / max(weights, 1e-9)))

    def _window_logits(self, window, real=1.0):
        """Logits of one window (already padded to the model's multiple)."""
        from ._saliency import as_head
        from .models.backbone import BuiltinBackbone
        clf, bb = self.classifier, self.backbone
        x = normalize_image(window, bb.mean, bb.std) if window.dtype == np.uint8 else \
            np.ascontiguousarray(window.transpose(2, 0, 1)[None], np.float32)
        names = list(self._token_layers)
        if clf is None and isinstance(bb, BuiltinBackbone):
            names += [n for n in [bb.blocks[-1], *bb.extra_outputs] if n not in names]
        maps = bb(x, names)
        if clf is not None:
            head = as_head(clf)
            feat = bb.head_features(maps[self._token_layers[-1]][0])
            pooled = feat.reshape(feat.shape[0], -1).mean(1)
            return head["weight"] @ pooled + (0 if head["bias"] is None else head["bias"])
        try:
            return bb.logits(maps, None)
        except KeyError:
            return None

    # -- ordering and colours ------------------------------------------------------
    def update_patches(self, images, min_size=None):
        """Recompute the patches (closest and largest cell of every concept) on `images`.

        min_size : minimum height and width, in pixels, of a cell to be the closest one. If no
            cell is large enough, the largest is used (with a warning); a concept with no cell
            at all gets an empty patch.
        """
        self._check_fitted()
        images = self._as_list(images)
        self._scan(images, None, make_progress(self.progress), min_size=min_size, patches_only=True)
        return self

    # -- explanations ----------------------------------------------------------------
    def predict(self, images, groups=None, uncertainty=False):
        """Explanations of `images` with the fitted concepts: a list of `ImageExplanation`s.

        groups : one label per image: use only that group's projector (when fitted with
            merge_groups=False). None: every projector scores every pixel, the closest wins.
        uncertainty : also compute the uncertainty map H(softmax(scores / T)) / log K
        """
        self._check_fitted()
        single = self._is_single(images)
        images = self._as_list(images)
        if groups is not None and self._merged:
            warnings.warn("This explainer merged its groups (merge_groups=True): groups are ignored.", stacklevel=2)
            groups = None
        if groups is not None and len(groups) != len(images):
            raise ValueError("groups must have one label per image.")
        self._check_memory([as_rgb(im).shape for im in images[:64]], uncertainty)
        report = make_progress(self.progress)
        K = len(self.palette)
        out = []
        for i, im in enumerate(images):
            rgb, net = self._prepare(im)
            engine = self._engine(net, uncertainty)
            engine.temperature = self.temperature_
            scorer, cols = self._stack, None
            if groups is not None:
                g = self._group_labels.index(groups[i])
                cols = np.nonzero(self._concept_group == g)[0]
                scorer = _Subset(self._stack, cols)
            clusters, unc = engine.assign(scorer, _identity)
            if cols is not None:
                clusters = cols[clusters].astype(np.int32)
            if clusters.shape != rgb.shape[:2]:
                H, W = rgb.shape[:2]
                ys = np.minimum((np.arange(H) * clusters.shape[0] / H).astype(int), clusters.shape[0] - 1)
                xs = np.minimum((np.arange(W) * clusters.shape[1] / W).astype(int), clusters.shape[1] - 1)
                clusters = clusters[ys][:, xs]
                unc = None if unc is None else np.clip(resample(unc, H, W, "linear"), 0, 1)
            out.append(ImageExplanation(np.ascontiguousarray(clusters, np.int32), unc, K))
            report((i + 1) / len(images), f"explaining {i + 1}/{len(images)}")
        return out[0] if single else out

    def fit_predict(self, images, groups=None, merge_groups=True, reorder="ward", uncertainty=False):
        """`fit`, then `predict` on the same images."""
        self.fit(images, groups, merge_groups, reorder)
        return self.predict(images, None if merge_groups else groups, uncertainty)

    def explain(self, images, uncertainty=False):
        """Explain images: with the fitted concepts after an explicit `fit`, otherwise by fitting
        afresh on these images (every call). One image gives one `ImageExplanation`, several a list."""
        if not self.fitted:
            self._fit(images, None, True, "ward")
        return self.predict(images, uncertainty=uncertainty)

    def segment(self, image):
        """Superpixels: `explain(image).segmentation()`."""
        return self.explain(image).segmentation()

    # -- saliency and summary ------------------------------------------------------------
    def saliency(self, images, classifier=None, target=None):
        """Concept-based saliency: the cosine between the deepest layer's features and the
        classifier's vector of the target class, averaged over NAVE regions from the deepest
        layer to the shallowest. One (H, W) float32 map per image (a list for several).

        classifier : as in the constructor (overrides it); None uses the constructor's, else
            the model's own head
        target : class index; None = the class the model predicts
        """
        from ._saliency import saliency as concept_saliency
        self.backbone
        single = self._is_single(images)
        out = []
        for im in self._as_list(images):
            rgb, net = self._prepare(im)
            H, W = net.shape[:2]
            if max(H, W) > 1024:                     # whole-image method: at most 1024 px, map resized back
                f = 1024 / max(H, W)
                h, w = max(1, int(H * f)), max(1, int(W * f))
                net = resize(net, h, w) if net.dtype == np.uint8 else resample(net, h, w, "linear")
            res = concept_saliency(net, self.backbone, self._token_layers, self.k,
                                   classifier=classifier if classifier is not None else self.classifier,
                                   target=target, upsample=self.upsample, normalize=self.normalize,
                                   scale=self.scale, clusterer=self.projector, alpha=self.bayes_alpha,
                                   iterations=self.bayes_iter, burn_in=self.bayes_burn_in,
                                   sample_size=self.sample_size, seed=self.seed)
            m = res.saliency
            if m.shape != rgb.shape[:2]:
                m = resample(m, rgb.shape[0], rgb.shape[1], self.upsample)
            out.append(np.clip(m, -1, 1).astype(np.float32))
        return out[0] if single else out

    def summarize(self, image, correlation=True, uncertainty=False, saliency=False, patches="closest",
                  classifier=None, target=None, label=None, saliency_range=(-1, 1)):
        """A figure in the style of the paper's Figure 1, for one image.

        Row 1: the image | its explanation (uncertainty as opacity on white if asked) | its saliency
        (if asked), all at the image's size.
        Then, for every block of 10 concepts: their correlation with the classes (if available and
        asked) and their patches ("closest" or "largest"), framed in the concept's colour.
        Returns the matplotlib Figure.
        saliency_range : (vmin, vmax) of the saliency colours (default (-1, 1): saliency is a
            cosine), or "auto": centred on 0, up to the item's largest |saliency|
        """
        from ._figure import summarize as draw_summary
        if patches not in KINDS:
            raise ValueError(f"patches must be one of {KINDS}.")
        from ._figure_sequence import concept_saliency, headline, saliency_range as sal_range
        exp = self.explain(image, uncertainty=uncertainty)
        rgb, net = self._prepare(image)
        need = saliency or (correlation and self.correlation is None)    # no correlation: mean saliency instead
        try:
            sal = self.saliency(image, classifier, target) if need else None
        except ValueError:                                                 # no classifier: no saliency
            sal = None
        mean_sal = concept_saliency(exp.explanation, sal, exp.K) \
            if (sal is not None and correlation and self.correlation is None) else None
        predicted = self._predict_class(rgb, net) if self._can_predict() else None
        title = headline(self.class_names, predicted, label)
        rng = sal_range(saliency_range, sal)
        return draw_summary(self, rgb, exp, sal if saliency else None, correlation, KINDS.index(patches), title,
                            mean_sal, rng)
