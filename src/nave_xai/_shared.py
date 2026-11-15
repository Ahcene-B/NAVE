# SPDX-License-Identifier: MIT
"""What NAVEImage, NAVESeries and NAVEText share: projectors per group, ordering and colours."""
import numpy as np

from ._summary import family_palette, prediction_groups
from .clustering import BayesianKMeans, ForeignClusterer, KMeans, as_clusterer


# --------------------------------------------------------------------------
# Several projectors (one per group), side by side
# --------------------------------------------------------------------------
class _Stack:
    """Fitted projectors with their concepts side by side, in a chosen order."""

    def __init__(self, projectors, sizes):
        self.projectors, self.sizes = projectors, list(sizes)
        self.order = np.arange(sum(self.sizes))

    def scores(self, X):
        return np.concatenate([p.scores(X) for p in self.projectors], axis=1)[:, self.order]

    @property
    def probabilistic(self):
        return all(isinstance(p, BayesianKMeans) or getattr(p, "kind", None) == "proba" for p in self.projectors)

    @property
    def centroid_based(self):
        """True when concepts are prototypes in composite-pixel space (k-means-like)."""
        return all(isinstance(p, (KMeans, BayesianKMeans)) or
                   (isinstance(p, ForeignClusterer) and p.kind in ("transform", "proba", "means", "predict"))
                   for p in self.projectors)

    @property
    def centroids_(self):
        parts = [getattr(p, "centroids_", None) for p in self.projectors]
        if any(c is None for c in parts):
            return None
        return np.concatenate([np.asarray(c, np.float32) for c in parts])[self.order]


class _Subset:
    """Scores of some concepts only (one group's)."""

    def __init__(self, scorer, cols):
        self.scorer, self.cols = scorer, np.asarray(cols)

    def scores(self, X):
        return self.scorer.scores(X)[:, self.cols]


def _identity(*_):
    return None


class _ConceptsBase:
    """Projectors, concept order and colours, common to every modality.

    Subclasses set `_PER_CONCEPT`: names of attributes indexed by concept (arrays or lists),
    permuted when concepts are renumbered."""

    _PER_CONCEPT = ()

    def _new_projector(self, seed_offset=0):
        spec = self.projector
        algo = as_clusterer(spec, self.k, self.seed + seed_offset, self.bayes_alpha, self.bayes_iter,
                            self.bayes_burn_in)
        if isinstance(spec, (str, type(None))) or seed_offset == 0:
            return algo
        from .clustering import fresh_copy
        return fresh_copy(algo, self.seed + seed_offset)

    def _set_projectors(self, projectors, sizes, labels, groups, merge_groups):
        self._stack = _Stack(projectors, sizes)
        self._group_labels = labels
        self._merged = merge_groups or groups is None
        self._concept_group = np.repeat(np.arange(len(labels)), sizes)
        self.prototypes = self._stack.centroids_
        self._weights = None if not all(isinstance(p, BayesianKMeans) for p in projectors) else \
            np.concatenate([p.weights_ * (np.sum(groups == g) / len(groups) if g is not None else 1.0)
                            for p, g in zip(projectors, labels)])

    def _finish_fit(self, groups, reorder):
        """Order the concepts (by group first, if any) and colour them."""
        K = sum(self._stack.sizes)
        if reorder is not None and self.prototypes is not None:
            if groups is not None:
                self._order_by(self._concept_group, reorder)
            else:
                from .hierarchy import concept_order
                self._apply_order(concept_order(self.prototypes, self._sizes, reorder)[0])
        self.palette = family_palette(self._concept_group, "lightness", 0.5) if groups is not None \
            else self._default_palette(K)

    @staticmethod
    def _default_palette(K):
        from ._draw import make_palette
        return make_palette(K, "Spectral")

    def _apply_order(self, order, keep_colors=True):
        order = np.asarray(order)
        self._stack.order = self._stack.order[order]
        self.prototypes = None if self.prototypes is None else self.prototypes[order]
        self._weights = None if self._weights is None else self._weights[order]
        self._sizes = self._sizes[order]
        self._concept_group = self._concept_group[order]
        self._proportions = self._proportions[:, order]
        for name in self._PER_CONCEPT:                 # modality-specific per-concept attributes
            value = getattr(self, name, None)
            if value is not None:
                setattr(self, name, value[order] if isinstance(value, np.ndarray) else [value[o] for o in order])
        if self.correlation is not None:
            self.correlation = self.correlation[order]
        if keep_colors and self.palette is not None:
            self.palette = self.palette[order]

    def _order_by(self, groups, method):
        """Concepts by group (in group order, -1 last), then by a dendrogram within each group."""
        from .hierarchy import concept_order
        groups = np.asarray(groups)
        order, new_groups = [], []
        for g in [g for g in dict.fromkeys(sorted(set(groups.tolist()))) if g >= 0] + [-1]:
            members = np.nonzero(groups == g)[0]
            if not len(members):
                continue
            sub = concept_order(self.prototypes[members], self._sizes[members], method)[0] \
                if method is not None else np.arange(len(members))
            order += members[sub].tolist()
            new_groups += [g] * len(members)
        self._apply_order(np.array(order))
        return np.array(new_groups)

    def reorder(self, method="ward", by=None, threshold=0.1, recolor=False, color_vary="lightness",
                color_spread=0.5):
        """Renumber the concepts. Changes this explainer; returns it.

        method : "ward", "average", "single" (dendrogram of the prototypes, weighted by concept
            size), or a list giving the new order (new id i = old id method[i])
        by : None (one dendrogram) or "classes": grouped under the class they correlate with most
            (classes in `classes` order, concepts below `threshold` unassigned and last), then a
            dendrogram within each class
        threshold : minimum correlation to join a class
        recolor : False (colours follow their concepts) or True (a new palette for the new order:
            with by="classes" one colour per class, shades within it; else the Spectral gradient)
        color_vary : "lightness", "saturation", "hue", or a list of these (shades within a class)
        color_spread : how far the shades go, 0 to 1
        """
        from ._draw import make_palette
        from .hierarchy import concept_order
        self._check_fitted()
        K = len(self.palette)
        if not isinstance(method, str) and method is not None:
            order = np.asarray(method)
            if sorted(order.tolist()) != list(range(K)):
                raise ValueError(f"A custom order must be a permutation of 0..{K - 1}.")
            self._apply_order(order)
            if recolor:
                self.palette = make_palette(K, "Spectral")
            return self
        if self.prototypes is None:
            raise ValueError("This projector has no prototypes to order.")
        if by is None:
            self._apply_order(concept_order(self.prototypes, self._sizes, method)[0])
            if recolor:
                self.palette = make_palette(K, "Spectral")
        elif by == "classes":
            if self.correlation is None:
                raise ValueError("reorder(by='classes') needs classes: fit with groups, or with a model that "
                                 "predicts classes (or classifier=), on at least two images.")
            groups = self._order_by(prediction_groups(self.correlation, threshold), method)
            if recolor:
                self.palette = family_palette(groups, color_vary, color_spread)
        else:
            raise ValueError("by must be None or 'classes'.")
        return self

    def set_palette(self, palette):
        """Concept colours: a (K, 3) uint8 array or a matplotlib colormap name ("tab20", "ordered", ...)."""
        from ._draw import make_palette
        self._check_fitted()
        K = len(self.palette)
        if isinstance(palette, str):
            self.palette = make_palette(K, palette)
        else:
            palette = np.asarray(palette, np.uint8)
            if palette.shape != (K, 3):
                raise ValueError(f"palette must have shape ({K}, 3).")
            self.palette = palette
        return self

    def _check_fitted(self):
        if self._stack is None:
            raise RuntimeError("Fit first: fit(images) (or explain(image)).")

