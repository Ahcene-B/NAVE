# SPDX-License-Identifier: MIT
"""`NAVESeries`: NAVE explanations of time-series encoders (1D CNNs, transformers).

    >>> from nave_xai import NAVESeries
    >>> explainer = NAVESeries(model, layers=["block3", "block4", "block5"], k=20).fit(series)
    >>> exp = explainer.explain(series[0])     # SeriesExplanation: the concept of every time step

Series are channels first, as in PyTorch: an (N, C, T) array, a list of (C, T_i) arrays, or one
(C, T) or (T,) series. One concept per time step, shared by all channels (the encoder mixes them).
"""
import numpy as np

from ._sequence import SequenceExplanation, _SequenceBase, runs
from .interp import resample
from .models.sequence import as_sequence_backbone

PROFILE_POINTS = 40
POSITION_BINS = 12


class SeriesExplanation(SequenceExplanation):
    """`explanation` (T,), `uncertainty` (T,) or None, `K`; `segments()` and `border()`."""


class NAVESeries(_SequenceBase):
    """NAVE explainer for time series.

    Parameters
    ----------
    model : a torch.nn.Module, an .onnx path, a function x (1, C, T) -> {layer: (C_b, t_b)}, or a
        SequenceBackbone
    layers : the layers to combine (module names or the function's keys), shallowest first
    k : number of concepts per projector
    multiple : inputs are padded (mirrored) to a multiple of this length (the model's downsampling)
    channels_last : PyTorch transformer layers output (N, t, C): set True
    transforms : None, or a function (or list) applied to each (C, T) series (e.g. normalisation)
    channel_names : names of the channels, for `summarize`
    window : None (whole series) or a window length in steps (windows every half window, every
        step from the window whose centre is nearest)
    upsample : "linear" (default), "nearest" or "cubic": deeper layers and scores to every step
    Other parameters as in `NAVEImage`.

    Attributes (after fitting)
    --------------------------------
    prototypes, palette, classes, correlation, fitted : as in `NAVEImage`
    snippets : for each concept, a list of up to n (series, start, end, (C, L) values, offset):
        the segments closest to the prototype, with context
    profiles : (K, C, 40, 3) the 10th, 50th and 90th percentiles of the concept's segments,
        resampled to 40 points, per channel
    positions : (K, 12) histogram of where segments sit (relative position in the series)
    durations : (K, 2) median and mean segment length, in steps
    """

    _PER_CONCEPT = ("snippets", "profiles", "positions", "durations")

    def __init__(self, model, layers, k=10, multiple=1, channels_last=False, projector=None, bayes_alpha=1.0,
                 bayes_iter=200, bayes_burn_in=100, upsample="linear", window=None, fit_on="all",
                 sample_size=20000, epochs=1, normalize=True, scale=True, uncertainty_temp=0.1, transforms=None,
                 classifier=None, class_names=None, channel_names=None, n_snippets=5, seed=0, device="cpu",
                 progress=False):
        self._init_common(layers, k, projector, bayes_alpha, bayes_iter, bayes_burn_in, upsample, window, fit_on,
                          sample_size, epochs, normalize, scale, uncertainty_temp, classifier, class_names, seed,
                          device, progress)
        self.model, self.multiple, self.channels_last = model, multiple, channels_last
        self.transforms, self.channel_names, self.n_snippets = transforms, channel_names, n_snippets
        self._backbone = None
        self.snippets = self.profiles = self.positions = self.durations = None

    @property
    def backbone(self):
        if self._backbone is None:
            self._backbone = as_sequence_backbone(self.model, self.layers, self.multiple, self.device,
                                                  self.channels_last)
            self._backbone.multiple = max(self._backbone.multiple, self.multiple)
            self.layers_ = list(self.layers)
        return self._backbone

    # -- inputs ---------------------------------------------------------------------------
    @staticmethod
    def _is_single(items):
        a = items if isinstance(items, np.ndarray) else None
        if hasattr(items, "detach"):
            a = items.detach().cpu().numpy()
        return a is not None and a.ndim in (1, 2)

    @classmethod
    def _as_list(cls, items):
        if hasattr(items, "detach"):
            items = items.detach().cpu().numpy()
        if isinstance(items, np.ndarray):
            if items.ndim == 3:
                return list(items)
            if items.ndim in (1, 2):
                return [items]
        return [np.asarray(i.detach().cpu().numpy() if hasattr(i, "detach") else i) for i in items]

    def _prepare(self, item):
        x = np.asarray(item, np.float32)
        if x.ndim == 1:
            x = x[None]
        if self.transforms is not None:
            for f in self.transforms if isinstance(self.transforms, (list, tuple)) else [self.transforms]:
                x = np.asarray(f(x), np.float32)
        return x

    @staticmethod
    def _length(x):
        return x.shape[1]

    @staticmethod
    def _slice(x, a, b):
        return x[:, a:b]

    def _run_backbone(self, x, n, names=None):
        pad = self._padded_length(n) - n
        if pad:
            x = np.pad(x, ((0, 0), (0, pad)), mode="reflect" if n > pad else "edge")
        return self.backbone(x[None], list(self.layers_) if names is None else list(names))

    def _explanation(self, x, labels, unc, K):
        return SeriesExplanation(labels, unc, K)

    def _own_head(self):
        clf = getattr(self.backbone, "model", None)
        if clf is None:
            return None
        from ._saliency import as_head
        for name in ("fc", "head", "classifier"):
            m = getattr(clf, name, None)
            if m is not None and hasattr(m, "weight"):
                return as_head(m)
        return None

    # -- per-concept examples -----------------------------------------------------------------
    def _new_trackers(self, K):
        rng = np.random.default_rng(self.seed)
        return dict(K=K, rng=rng, closest=[[] for _ in range(K)], profiles=[[] for _ in range(K)],
                    seen=np.zeros(K, int), positions=np.zeros((K, POSITION_BINS)), lengths=[[] for _ in range(K)])

    def _track(self, tr, i, x, labels, dist):
        T = len(labels)
        for a, b, c in runs(labels):
            tr["lengths"][c].append(b - a)
            tr["positions"][c, min(POSITION_BINS - 1, int((a + b) / 2 / T * POSITION_BINS))] += 1
            d = float(dist[a:b].mean())
            ctx = max(3, (b - a) // 2)
            lo, hi = max(0, a - ctx), min(T, b + ctx)
            cl = tr["closest"][c]
            if b - a >= 2 and (len(cl) < self.n_snippets or d < cl[-1][0]):
                cl.append((d, i, a, b, x[:, lo:hi].copy(), a - lo))
                cl.sort(key=lambda o: o[0])
                del cl[self.n_snippets:]
            if b - a >= 2:                                        # reservoir of 300 resampled segments
                tr["seen"][c] += 1
                p = resample(x[:, a:b], x.shape[0], PROFILE_POINTS, "linear")
                if len(tr["profiles"][c]) < 300:
                    tr["profiles"][c].append(p)
                else:
                    j = tr["rng"].integers(0, tr["seen"][c])
                    if j < 300:
                        tr["profiles"][c][j] = p

    def _finish_trackers(self, tr):
        K = tr["K"]
        self.snippets = [[(i, a, b, v, off) for _, i, a, b, v, off in tr["closest"][k]] for k in range(K)]
        C = next((p[0].shape[0] for p in tr["profiles"] if p), 1)
        prof = np.full((K, C, PROFILE_POINTS, 3), np.nan, np.float32)
        for k in range(K):
            if tr["profiles"][k]:
                prof[k] = np.moveaxis(np.percentile(np.stack(tr["profiles"][k]), [10, 50, 90], axis=0), 0, -1)
        self.profiles = prof
        self.positions = tr["positions"]
        self.durations = np.array([[np.median(lv), np.mean(lv)] if lv else [0.0, 0.0] for lv in tr["lengths"]])

    def update_snippets(self, series):
        """Recompute snippets, profiles, positions and durations on `series`."""
        self._check_fitted()
        self._scan([self._prepare(s) for s in self._as_list(series)], None, lambda *a: None, examples_only=True)
        return self

    # -- summary ---------------------------------------------------------------------------------
    def summarize(self, series, uncertainty=False, saliency=False, span=None, classifier=None, target=None,
                  label=None, saliency_range=(-1, 1)):
        """The figure for one series: one row per channel (the signal over the segmentation),
        an uncertainty row (if asked; inferno reversed: bright = certain, black = uncertain),
        a saliency row (if asked), the correlation of the
        concepts with the classes, and per concept: position & duration, snippets, profile
        (one band per channel), 10 concepts per row. span=(start, end) zooms on a part.
        label : the series' true class, shown with the model's prediction (when it predicts).
        saliency_range : (vmin, vmax) of the saliency colours (default (-1, 1): saliency is a
            cosine), or "auto": centred on 0, up to the item's largest |saliency|"""
        from ._figure_sequence import concept_saliency, headline, saliency_range as sal_range, series_summary
        x = self._prepare(series)
        exp = self.explain(series, uncertainty=uncertainty)
        try:
            sal = self.saliency(series, classifier, target) if (saliency or self.correlation is None) else None
        except ValueError:                                                 # no classifier: no saliency
            sal = None
        mean_sal = concept_saliency(exp.explanation, sal, exp.K) if (sal is not None and self.correlation is None) else None
        return series_summary(self, x, exp, sal if saliency else None, span,
                              headline(self.class_names, self._predict_class(x), label), mean_sal,
                              sal_range(saliency_range, sal))
