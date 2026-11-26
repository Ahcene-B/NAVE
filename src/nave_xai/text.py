# SPDX-License-Identifier: MIT
"""`NAVEText`: NAVE explanations of text encoders, one concept per token.

    >>> from nave_xai import NAVEText
    >>> explainer = NAVEText("spacy").fit(documents)     # pip install "nave-xai[text]"
    >>> exp = explainer.explain("Oil prices rose after OPEC cut production.")
    >>> list(zip(exp.tokens, exp.explanation))

Inputs: a string is one document, a list of strings several documents; an already tokenized
document is a list of tokens, so several of them are a list of lists.
"""
import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from ._sequence import SequenceExplanation, _SequenceBase, runs
from .models.sequence import FunctionSequenceBackbone, as_sequence_backbone

SPACY_MODEL = "en_core_web_sm"


@dataclass
class TextExplanation(SequenceExplanation):
    """`tokens`, `explanation` (n_tokens,) the concept of every token, `uncertainty` or None, `K`;
    `segments()` (runs of tokens with the same concept) and `border()`."""

    tokens: list = field(default_factory=list)


def spacy_backbone(name=SPACY_MODEL):
    """spaCy's tok2vec as a backbone: hashed embeddings, then four residual window-CNN blocks
    ("block1" ... "block4", 96 channels per token). Reproduces spaCy's own output exactly."""
    try:
        import spacy
    except ImportError as err:
        raise ImportError('model="spacy" needs spaCy: pip install "nave-xai[text]"') from err
    try:
        nlp = spacy.load(name)
    except OSError as err:
        raise OSError(f"spaCy's {name} is not installed: python -m spacy download {name} "
                      "(about 12 MB, from GitHub)") from err
    from spacy.tokens import Doc
    t2v = nlp.get_pipe("tok2vec").model
    embed, encode = t2v.layers
    blocks, pad = encode.layers[0].layers, encode.attrs["pad"]

    def features(tokens):
        x = embed.predict([Doc(nlp.vocab, words=list(tokens) or [""])])[0]
        h = np.pad(x, ((pad, pad), (0, 0)))
        out = {}
        for i, b in enumerate(blocks):
            h = b.predict(h)
            out[f"block{i + 1}"] = h[pad:len(h) - pad].T.copy()
        return out

    backbone = FunctionSequenceBackbone(features, [f"block{i + 1}" for i in range(len(blocks))])
    backbone.tokenizer = lambda text: [t.text for t in nlp.make_doc(text)]
    return backbone


class NAVEText(_SequenceBase):
    """NAVE explainer for text.

    Parameters
    ----------
    model : "spacy" (spaCy's tok2vec, needs `pip install "nave-xai[text]"`), a function
        tokens -> {layer: (C_b, n_tokens), optional "logits"}, or a SequenceBackbone. For PyTorch
        or Hugging Face models, wrap the model in such a function (tokens to ids inside).
    layers : layers to combine, shallowest first (spaCy: default "block2", "block3", "block4")
    k : number of concepts per projector (20 by default: text needs more than images)
    tokenizer : function text -> tokens (spaCy's with model="spacy", else words and punctuation)
    window : None (whole document) or a window length in tokens
    n_words : characteristic words shown per concept
    Other parameters as in `NAVEImage`.

    Attributes (after fitting)
    --------------------------------
    prototypes, palette, classes, correlation, fitted : as in `NAVEImage`
    span_counts : (K,) number of spans (runs of consecutive tokens) of each concept
    mean_runs : (K,) mean number of consecutive tokens with the concept
    words : for each concept, its characteristic words (frequency in the concept relative to overall)
    spans : for each concept, its closest spans to the prototype: (tokens with context, start, end)
    """

    _PER_CONCEPT = ("span_counts", "mean_runs", "words", "spans")

    def __init__(self, model="spacy", layers=None, k=20, tokenizer=None, projector=None, bayes_alpha=1.0,
                 bayes_iter=200, bayes_burn_in=100, upsample="nearest", window=None, fit_on="all",
                 sample_size=20000, epochs=1, normalize=True, scale=True, uncertainty_temp=0.1, classifier=None,
                 class_names=None, n_words=6, seed=0, device="cpu", progress=False):
        self._init_common(layers, k, projector, bayes_alpha, bayes_iter, bayes_burn_in, upsample, window, fit_on,
                          sample_size, epochs, normalize, scale, uncertainty_temp, classifier, class_names, seed,
                          device, progress)
        self.model, self.tokenizer, self.n_words = model, tokenizer, n_words
        self._backbone = None
        self.span_counts = self.mean_runs = self.words = self.spans = None

    @property
    def backbone(self):
        if self._backbone is None:
            if isinstance(self.model, str) and self.model == "spacy":
                self._backbone = spacy_backbone()
                self.layers_ = list(self.layers or ["block2", "block3", "block4"])
            else:
                self._backbone = as_sequence_backbone(self.model, self.layers, 1, self.device)
                self.layers_ = list(self.layers)
        return self._backbone

    # -- inputs ----------------------------------------------------------------------------------
    @staticmethod
    def _is_single(items):
        return isinstance(items, str)

    @staticmethod
    def _as_list(items):
        return [items] if isinstance(items, str) else list(items)

    def _tokenize(self, text):
        if self.tokenizer is not None:
            return list(self.tokenizer(text))
        tok = getattr(self.backbone, "tokenizer", None)
        return tok(text) if tok is not None else re.findall(r"\w+(?:['.-]\w+)*|[^\w\s]", text)

    def _prepare(self, item):
        return self._tokenize(item) if isinstance(item, str) else list(item)

    @staticmethod
    def _length(x):
        return len(x)

    @staticmethod
    def _slice(x, a, b):
        return x[a:b]

    def _run_backbone(self, tokens, n, names=None):
        return self.backbone(tokens, list(self.layers_) if names is None else list(names))

    def _padded_length(self, n):
        return n

    def _explanation(self, x, labels, unc, K):
        return TextExplanation(labels, unc, K, list(x))

    # -- per-concept examples ----------------------------------------------------------------------
    def _new_trackers(self, K):
        return dict(K=K, counts=np.zeros(K, int), lengths=[[] for _ in range(K)], words=[Counter() for _ in range(K)],
                    total=Counter(), closest=[[] for _ in range(K)])

    def _track(self, tr, i, x, labels, dist):
        for t, c in zip(x, labels):
            w = t.lower()
            tr["words"][c][w] += 1
            tr["total"][w] += 1
        for a, b, c in runs(labels):
            tr["counts"][c] += 1
            tr["lengths"][c].append(b - a)
            d = float(dist[a:b].mean())
            cl = tr["closest"][c]
            if len(cl) < 2 or d < cl[-1][0]:
                lo, hi = max(0, a - 6), min(len(x), b + 6)
                cl.append((d, list(x[lo:hi]), a - lo, b - lo))
                cl.sort(key=lambda o: o[0])
                del cl[2:]

    def _finish_trackers(self, tr):
        K, n_all = tr["K"], max(sum(tr["total"].values()), 1)
        self.span_counts = tr["counts"]
        self.mean_runs = np.array([np.mean(v) if v else 0.0 for v in tr["lengths"]])
        words = []
        for k in range(K):
            n_k = max(sum(tr["words"][k].values()), 1)
            lift = {w: (n / n_k) / (tr["total"][w] / n_all) for w, n in tr["words"][k].items() if n >= 3}
            words.append(sorted(lift, key=lambda w: (-lift[w], -tr["words"][k][w]))[:self.n_words])
        self.words = words
        self.spans = [[(toks, a, b) for _, toks, a, b in tr["closest"][k]] for k in range(K)]

    def update_spans(self, texts):
        """Recompute span counts, mean runs, characteristic words and closest spans on `texts`."""
        self._check_fitted()
        self._scan([self._prepare(t) for t in self._as_list(texts)], None, lambda *a: None, examples_only=True)
        return self

    # -- summary -----------------------------------------------------------------------------------
    def summarize(self, text, uncertainty=True, saliency=False, classifier=None, target=None, label=None,
                  title=None, saliency_range=(-1, 1)):
        """The figure for one document: the text wrapped, each token on its concept's colour,
        the background's opacity = certainty; with saliency=True, the font size shows the
        saliency's magnitude and black / white its sign. Then the correlation of the concepts
        with the classes, and one row per concept: colour | spans and mean run | characteristic
        words | two closest spans in context.
        label : the document's true class, shown with the model's prediction (when it predicts).
        saliency_range : (vmin, vmax) of the saliency colours (default (-1, 1): saliency is a
            cosine), or "auto": centred on 0, up to the item's largest |saliency|"""
        from ._figure_sequence import headline, text_summary
        from ._figure_sequence import concept_saliency, saliency_range as sal_range
        tokens = self._prepare(text)
        exp = self.explain([tokens], uncertainty=uncertainty)[0]
        try:
            sal = self.saliency([tokens], classifier, target)[0] if (saliency or self.correlation is None) else None
        except ValueError:                                                 # no classifier: no saliency
            sal = None
        mean_sal = concept_saliency(exp.explanation, sal, exp.K) if (sal is not None and self.correlation is None) else None
        head = headline(self.class_names, self._predict_class(tokens), label)
        return text_summary(self, tokens, exp, sal if saliency else None,
                            title or ("Document" + (f"   ({head})" if head else "")), mean_sal,
                            sal_range(saliency_range, sal))
