# SPDX-License-Identifier: MIT
"""The engine shared by NAVESeries and NAVEText: one concept per step (time step or token).

A layer's output (C, t) is handled as a feature map of height 1, so Algorithm 1 (upsample,
normalise, scale, concatenate) is the package's own CompositeFeatures. Long sequences are cut
into windows of W steps every W/2 steps; every step comes from the window whose centre is
nearest. Concept scores are interpolated from the feature grid to every step.
"""
import math
import warnings
from dataclasses import dataclass

import numpy as np

from . import memory
from ._engine import normalised_entropy
from ._shared import _ConceptsBase, _Subset
from ._summary import classes_by_frequency, correlation as correlate
from .clustering import BayesianKMeans, KMeans, is_fitted
from .composite import CompositeFeatures
from .interp import check_mode, resample


@dataclass
class SequenceExplanation:
    """The explanation of one sequence: `explanation` (T,) int32, the concept of every step;
    `uncertainty` (T,) float32 in [0, 1] or None; `K` the number of concepts."""

    explanation: np.ndarray
    uncertainty: np.ndarray = None
    K: int = 0

    def segments(self):
        """(T,) int32 run ids: consecutive steps with the same concept share an id."""
        e = self.explanation
        return np.concatenate([[0], np.cumsum(e[1:] != e[:-1])]).astype(np.int32) if len(e) else e

    def border(self):
        """(T,) int32, 1 at the first step of every new segment (change points), else 0."""
        e = self.explanation
        out = np.zeros(len(e), np.int32)
        out[1:] = e[1:] != e[:-1]
        return out


def runs(labels):
    """[(start, end, concept)] of a label sequence."""
    if not len(labels):
        return []
    change = np.r_[0, np.nonzero(np.diff(labels))[0] + 1, len(labels)]
    return [(int(a), int(b), int(labels[a])) for a, b in zip(change[:-1], change[1:])]


class _SequenceBase(_ConceptsBase):
    """fit / predict / explain / saliency over sequences. Subclasses provide `_prepare`,
    `_length`, `_slice`, `_run_backbone`, and the per-concept trackers."""

    def _init_common(self, layers, k, projector, bayes_alpha, bayes_iter, bayes_burn_in, upsample, window,
                     fit_on, sample_size, epochs, normalize, scale, uncertainty_temp, classifier, class_names,
                     seed, device, progress):
        check_mode(upsample)
        if fit_on not in ("all", "sample"):
            raise ValueError("fit_on must be 'all' or 'sample'.")
        self.layers, self.k, self.projector = layers, k, projector
        self.bayes_alpha, self.bayes_iter, self.bayes_burn_in = bayes_alpha, bayes_iter, bayes_burn_in
        self.upsample, self.window, self.fit_on, self.sample_size, self.epochs = upsample, window, fit_on, sample_size, epochs
        self.normalize, self.scale, self.uncertainty_temp = normalize, scale, uncertainty_temp
        self.classifier, self.class_names = classifier, class_names
        self.seed, self.device, self.progress = seed, device, progress
        self.fitted = False
        self.prototypes = self.palette = self.classes = self.correlation = None
        self._stack = None

    # -- features ------------------------------------------------------------------
    def _composite(self, maps, names):
        comp = CompositeFeatures([np.asarray(maps[n], np.float32)[:, None, :] for n in names],
                                 self.upsample, self.normalize, self.scale)
        return comp.full()[0], comp

    def _windows(self, T):
        """[(start, end, keep_start, keep_end)]: windows of W steps every W/2 (rounded to the
        model's multiple); each keeps the steps whose nearest window centre is its own."""
        W, m = self.window, max(1, self.backbone.multiple)
        if W is None or T <= W:
            return [(0, T, 0, T)]
        W = int(W)
        margin = max(m, (W // 4) // m * m)
        step = W - 2 * margin
        out, s = [], 0
        while True:
            keep0 = s + (margin if s > 0 else 0)
            start = s
            end = min(T, start + W)
            last = end >= T
            keep1 = T if last else start + W - margin
            out.append((start, end, keep0, keep1))
            if last:
                break
            s += step
        return out

    def _features(self, x):
        """Composite rows covering the whole sequence (R, D) and the steps per row."""
        T = self._length(x)
        parts, stride = [], None
        for start, end, k0, k1 in self._windows(T):
            maps = self._run_backbone(self._slice(x, start, end), end - start)
            Z, _ = self._composite(maps, self.layers_)
            n = end - start
            st = self._padded_length(n) / len(Z)
            stride = st if stride is None else stride
            r0, r1 = int(round((k0 - start) / st)), int(math.ceil((k1 - start) / st))
            parts.append(Z[r0:min(r1, len(Z))])
        Z = np.concatenate(parts)
        return Z, stride, T

    def _padded_length(self, n):
        m = max(1, self.backbone.multiple)
        return int(math.ceil(n / m) * m)

    def _to_steps(self, values, stride, T):
        """(R, K) values on the feature grid -> (T, K) at every step."""
        R = len(values)
        if R == T:
            return values
        mode = self.upsample
        full = resample(np.ascontiguousarray(values.T), values.shape[1], int(round(R * stride)), mode).T
        if len(full) < T:
            full = np.concatenate([full, np.repeat(full[-1:], T - len(full), 0)])
        return full[:T]

    # -- fitting -----------------------------------------------------------------------
    def fit(self, items, groups=None, merge_groups=True, reorder="ward"):
        """Extract concepts from `items`; see the class docstring for accepted inputs.

        groups : one label per item: one projector per group, and the groups become `classes`
        merge_groups : merge the groups' projectors into one (`predict` then ignores groups)
        reorder : "ward", "average", "single" or None: order of the concept ids (by group first)
        """
        self._fit(items, groups, merge_groups, reorder)
        self.fitted = True
        return self

    def _fit(self, items, groups, merge_groups, reorder):
        from ._progress import make_progress, span
        items = self._as_list(items)
        report = make_progress(self.progress)
        self.backbone
        if groups is not None:
            groups = np.asarray(groups)
            if len(groups) != len(items):
                raise ValueError("groups must have one label per item.")
            labels = list(dict.fromkeys(groups.tolist()))
        else:
            labels = [None]
        prepared = [self._prepare(it) for it in items]
        projectors, sizes = [], []
        for i, g in enumerate(labels):
            members = prepared if g is None else [p for p, gi in zip(prepared, groups) if gi == g]
            proj = self._new_projector(i)
            Z = self._fit_projector(proj, members)
            sizes.append(proj.scores(Z[:1]).shape[1])
            projectors.append(proj)
            report(0.7 * (i + 1) / len(labels), "concepts")
        self._set_projectors(projectors, sizes, labels, groups, merge_groups)
        self._scan(prepared, groups, span(report, 0.7, 0.97))
        self._finish_fit(groups, reorder)
        report(1.0, "fitted")

    def _fit_projector(self, proj, prepared):
        rows = [self._features(x)[0] for x in prepared]
        if is_fitted(proj):
            return rows[0][:16]
        total, dim = sum(len(r) for r in rows), rows[0].shape[1]
        free = memory.available()
        X = np.concatenate(rows)
        if isinstance(proj, BayesianKMeans) and self.fit_on == "all" and len(X) > self.sample_size:
            warnings.warn(f"projector='bayesian' with fit_on='all': Gibbs sampling on every step would take "
                          f"very long; fitted on {self.sample_size} sampled steps.", stacklevel=4)
            X = X[np.random.default_rng(self.seed).choice(len(X), self.sample_size, replace=False)]
        elif self.fit_on == "sample" and len(X) > self.sample_size:
            X = X[np.random.default_rng(self.seed).choice(len(X), self.sample_size, replace=False)]
        elif self.fit_on == "all" and free is not None and 4 * total * dim > 0.5 * free:
            if isinstance(proj, KMeans):
                warnings.warn(f"fit_on='all': {total} steps need {4 * total * dim / memory.GB:.1f} GB; streamed "
                              f"through mini-batch k-means instead of exact k-means.", stacklevel=4)
                rng = np.random.default_rng(self.seed)
                proj.fit(X[rng.choice(len(X), min(len(X), self.sample_size), replace=False)])
                for _ in range(self.epochs):
                    for chunk in np.array_split(X[rng.permutation(len(X))], max(1, len(X) // 4096)):
                        proj.partial_fit(chunk)
                return X[:16]
            warnings.warn(f"fit_on='all': too many steps for memory; fitted on {self.sample_size} sampled steps.",
                          stacklevel=4)
            X = X[np.random.default_rng(self.seed).choice(len(X), self.sample_size, replace=False)]
        proj.fit(X)
        return X[:16]

    # -- one pass with the fitted concepts ------------------------------------------------
    def _assign(self, x, scorer):
        Z, stride, T = self._features(x)
        S = scorer.scores(Z)
        lab_rows = S.argmax(1)
        if self.prototypes is not None and self._stack.centroid_based:
            d_rows = np.sqrt(((Z - self.prototypes[lab_rows]) ** 2).sum(1))
        else:
            d_rows = -S.max(1)
        S_steps = self._to_steps(S, stride, T)
        dist = self._to_steps(d_rows[:, None], stride, T)[:, 0]
        return S_steps, S_steps.argmax(1).astype(np.int32), dist, S

    def _scan(self, prepared, groups, report, examples_only=False):
        K = sum(self._stack.sizes)
        trackers = self._new_trackers(K)
        sizes, gaps, proportions, predictions = np.zeros(K), [], [], []
        predict = groups is None and not examples_only and self._can_predict()
        for i, x in enumerate(prepared):
            S, labels, dist, S_rows = self._assign(x, self._stack)
            counts = np.bincount(labels, minlength=K)
            sizes += counts
            proportions.append(counts / max(counts.sum(), 1))
            if len(gaps) < 8 and S_rows.shape[1] > 1:
                top2 = np.partition(S_rows, -2, axis=1)[:, -2:]
                gaps.append(float(np.median(top2[:, 1] - top2[:, 0])))
            self._track(trackers, i, x, labels, dist)
            if predict:
                predictions.append(self._predict_class(x))
            report((i + 1) / len(prepared), f"examples and classes {i + 1}/{len(prepared)}")
        self._finish_trackers(trackers)
        if examples_only:
            return
        self._sizes = sizes
        self.temperature_ = float(self.uncertainty_temp) if self.uncertainty_temp != "auto" else (
            1.0 if self._stack.probabilistic or not gaps else max(float(np.median(gaps)), 1e-6))
        self._proportions = np.array(proportions)
        labels = groups if groups is not None else (
            np.array(predictions) if predict and all(p is not None for p in predictions) else None)
        if predict and labels is None and self.classifier is not None:
            warnings.warn("The classifier gave no class prediction: no classes and correlation.", stacklevel=4)
        self.classes = None if labels is None else classes_by_frequency(labels)
        several = labels is not None and len(prepared) >= 2 and len(set(np.asarray(labels).tolist())) >= 2
        self.correlation = correlate(self._proportions, labels, self.classes) if several else None

    # -- classes ------------------------------------------------------------------------------
    def _can_predict(self):
        return True

    def _logits(self, x):
        """Class scores of one sequence: the model's own, or the classifier's on the deepest
        layer averaged over the sequence (windows weighted by their kept length)."""
        from ._saliency import as_head
        clf = self.classifier
        if clf is not None and callable(clf) and not hasattr(clf, "weight") \
                and not isinstance(clf, (np.ndarray, list, tuple, dict)):
            return np.asarray(clf(self._raw(x)), np.float32)
        T, total, weight = self._length(x), None, 0.0
        for start, end, k0, k1 in self._windows(T):
            names = [self.layers_[-1]]
            maps = self._run_backbone(self._slice(x, start, end), end - start, names)
            if clf is None:
                if "logits" not in maps:
                    return None
                logits = np.asarray(maps["logits"], np.float32)
            else:
                head = as_head(clf)
                pooled = np.asarray(maps[self.layers_[-1]], np.float32).mean(1)
                logits = head["weight"] @ pooled + (0 if head["bias"] is None else head["bias"])
            w = (k1 - k0) / max(T, 1)
            total = logits * w if total is None else total + logits * w
            weight += w
        return total / max(weight, 1e-9)

    def _predict_class(self, x):
        logits = self._logits(x)
        if logits is None:
            return None
        return int(logits) if np.ndim(logits) == 0 else int(np.argmax(logits))

    def _raw(self, x):
        return x

    # -- explanations ---------------------------------------------------------------------------
    def predict(self, items, groups=None, uncertainty=False):
        """Explanations with the fitted concepts: one for one item, a list for several.

        groups : one label per item: only that group's concepts (when fitted with
            merge_groups=False)
        uncertainty : also compute H(softmax(scores / T)) / log K at every step
        """
        self._check_fitted()
        single = self._is_single(items)
        items = self._as_list(items)
        if groups is not None and self._merged:
            warnings.warn("This explainer merged its groups (merge_groups=True): groups are ignored.", stacklevel=2)
            groups = None
        K = len(self.palette)
        out = []
        for i, it in enumerate(items):
            x = self._prepare(it)
            scorer, cols = self._stack, None
            if groups is not None:
                cols = np.nonzero(self._concept_group == self._group_labels.index(groups[i]))[0]
                scorer = _Subset(self._stack, cols)
            S, labels, _, _ = self._assign(x, scorer)
            unc = normalised_entropy(S, self.temperature_) if uncertainty else None
            if cols is not None:
                labels = cols[labels].astype(np.int32)
            out.append(self._explanation(x, labels, unc, K))
        return out[0] if single else out

    def fit_predict(self, items, groups=None, merge_groups=True, reorder="ward", uncertainty=False):
        """`fit`, then `predict` on the same items."""
        self.fit(items, groups, merge_groups, reorder)
        return self.predict(items, None if merge_groups else groups, uncertainty)

    def explain(self, items, uncertainty=False):
        """With the fitted concepts after an explicit `fit`, otherwise fitting afresh on these items
        (every call). One item gives one explanation, several a list."""
        if not self.fitted:
            self._fit(items, None, True, "ward")
        return self.predict(items, uncertainty=uncertainty)

    def segment(self, item):
        """`explain(item).segments()`."""
        return self.explain(item).segments()

    # -- saliency -------------------------------------------------------------------------------
    def saliency(self, items, classifier=None, target=None):
        """Concept-based saliency: cosine between the deepest layer and the class vector, averaged
        over NAVE runs from the deepest layer to the shallowest. One (T,) map per item."""
        from ._saliency import as_head
        single = self._is_single(items)
        out = []
        for it in self._as_list(items):
            x = self._prepare(it)
            T = self._length(x)
            maps = self._run_backbone(x, T, list(self.layers_))
            head = as_head(classifier if classifier is not None else self.classifier) \
                if (classifier is not None or self.classifier is not None) else None
            if head is None:
                head = self._own_head()
            if head is None:
                raise ValueError("saliency needs a classifier: pass classifier= (a torch.nn.Linear, a weight "
                                 "matrix or a (weight, bias) pair) for the deepest layer.")
            t = target
            if t is None:
                logits = self._logits(x)
                t = int(np.argmax(logits)) if logits is not None else 0
            deep = np.asarray(maps[self.layers_[-1]], np.float32)
            w = np.asarray(head["weight"], np.float32)[t]
            current = (w @ deep) / (np.linalg.norm(deep, axis=0) * np.linalg.norm(w) + 1e-12)
            for j in range(len(self.layers_) - 1, -1, -1):
                Z, _ = self._composite(maps, self.layers_[j:])
                if len(current) != len(Z):
                    current = resample(current[None], 1, len(Z), "linear")[0]
                lab = KMeans(min(self.k, len(Z)), seed=self.seed).fit(Z).predict(Z)
                for a, b, _ in runs(lab):
                    current[a:b] = current[a:b].mean()
            stride = self._padded_length(T) / len(current)
            out.append(self._to_steps(current[:, None], stride, T)[:, 0].astype(np.float32))
        return out[0] if single else out

    def _own_head(self):
        return None
