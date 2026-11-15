import numpy as np
import pytest

from nave_xai import NAVESeries, NAVEText, SeriesExplanation, TextExplanation

RNG = np.random.default_rng(0)
A1, A2, A3 = RNG.normal(size=(8, 3)), RNG.normal(size=(12, 8)), RNG.normal(size=(16, 12))
HEAD = RNG.normal(size=(2, 16))


def smooth(h, k=3):
    return np.stack([np.convolve(r, np.ones(k) / k, mode="same") for r in h])


def series_model(x):                      # a random 1D CNN: (1, C, T) -> blocks at strides 1, 2, 4
    h1 = np.tanh(A1 @ x[0])
    h2 = np.tanh(A2 @ smooth(h1))[:, ::2]
    h3 = np.tanh(A3 @ smooth(h2))[:, ::2]
    return {"b1": h1, "b2": h2, "b3": h3, "logits": HEAD @ h3.mean(1)}


def make_series(n=12, T=64):
    t = np.linspace(0, 4 * np.pi, T)
    out, y = [], []
    for i in range(n):
        f = 1 + (i % 2)
        out.append(np.stack([np.sin(f * t + i), np.cos(f * t), RNG.normal(0, 0.1, T)]).astype(np.float32))
        y.append(i % 2)
    return np.stack(out), np.array(y)


def test_series_fit_predict():
    X, y = make_series()
    ex = NAVESeries(series_model, layers=["b1", "b2", "b3"], k=5, multiple=4).fit(X)
    assert ex.fitted and ex.prototypes.shape[0] == 5 and ex.palette.shape == (5, 3)
    assert ex.correlation.shape[0] == 5                               # classes from the model's logits
    assert ex.profiles.shape == (5, 3, 40, 3) and ex.positions.shape == (5, 12) and len(ex.snippets) == 5
    e = ex.predict(X[0], uncertainty=True)
    assert isinstance(e, SeriesExplanation) and e.explanation.shape == (64,) and e.uncertainty.shape == (64,)
    assert len(ex.predict(X[:3])) == 3
    assert ex._prepare(X[0][0]).shape == (1, 64) and ex._is_single(X[0][0])          # univariate: one channel
    assert e.segments().max() + 1 == e.border().sum() + 1


def test_series_windows_groups_reorder():
    X, y = make_series()
    ex = NAVESeries(series_model, layers=["b1", "b2", "b3"], k=3, multiple=4, window=32)
    ex.backbone
    w = ex._windows(64)
    assert w[0][2] == 0 and w[-1][3] == 64 and all(w[i][3] == w[i + 1][2] for i in range(len(w) - 1))
    ex.fit(X, groups=y, merge_groups=False)
    assert len(ex.palette) == 6 and list(ex.classes) == [0, 1]
    only = ex.predict(X[0], groups=[1]).explanation
    assert set(np.unique(only)) <= set(np.nonzero(ex._concept_group == 1)[0])
    before = ex.predict(X[1]).explanation
    ex.reorder(by="classes", recolor=True)
    assert ex.profiles.shape[0] == 6 and len(ex.predict(X[1]).explanation) == len(before)


def test_series_saliency_summary():
    import matplotlib
    matplotlib.use("Agg")
    X, _ = make_series()
    ex = NAVESeries(series_model, layers=["b1", "b2", "b3"], k=4, multiple=4, classifier=HEAD,
                    channel_names=["sin", "cos", "noise"]).fit(X)
    s = ex.saliency(X[0], target=1)
    assert s.shape == (64,)
    fig = ex.summarize(X[0], uncertainty=True, saliency=True, span=(8, 56))
    assert len(fig.axes) > 10


VOCAB = {}


def text_model(tokens):                   # random embeddings, two smoothing blocks
    vecs = [VOCAB.setdefault(t.lower(), np.random.default_rng(abs(hash(t.lower())) % 2 ** 32).normal(size=8))
            for t in tokens]
    h1 = np.stack(vecs, 1) if tokens else np.zeros((8, 0))
    h2 = np.tanh(smooth(h1))
    h3 = np.tanh(smooth(h2, 5))
    return {"b1": h1, "b2": h2, "b3": h3, "logits": np.array([h3[0].mean(), h3[1].mean()])}


DOCS = ["Oil prices rose after OPEC cut production .", "The company reported higher quarterly earnings .",
        "Crude oil exports fell sharply in March .", "Net profit rose to 12 mln dlrs from 9 mln .",
        "OPEC ministers will meet on oil output .", "Shares rose after the earnings report ."] * 3


def test_text_fit_explain():
    ex = NAVEText(text_model, layers=["b1", "b2", "b3"], k=6).fit(DOCS)
    e = ex.explain("Oil output rose .")
    assert isinstance(e, TextExplanation) and e.tokens == ["Oil", "output", "rose", "."]
    assert e.explanation.shape == (4,) and e.K == 6
    assert len(ex.words) == 6 and len(ex.spans) == 6 and ex.mean_runs.shape == (6,)
    assert len(ex.explain(["one doc .", "two docs ."])) == 2
    tok = ex.explain([["already", "tokenized"]])
    assert tok[0].tokens == ["already", "tokenized"]


def test_text_groups_summary():
    import matplotlib
    matplotlib.use("Agg")
    y = [0, 1, 0, 1, 0, 1] * 3
    ex = NAVEText(text_model, layers=["b1", "b2", "b3"], k=3, class_names=["crude", "earn"]).fit(DOCS, groups=y)
    assert len(ex.palette) == 6 and ex.correlation.shape == (6, 2)
    ex.reorder(by="classes", recolor=True)
    fig = ex.summarize(DOCS[0], uncertainty=True, saliency=True, classifier=np.eye(2, 8))
    assert fig is not None


def test_text_spacy():
    spacy = pytest.importorskip("spacy")
    try:
        spacy.load("en_core_web_sm")
    except OSError:
        pytest.skip("en_core_web_sm not installed")
    ex = NAVEText("spacy", k=4).fit(DOCS[:6])
    assert ex.layers_ == ["block2", "block3", "block4"]
    assert ex.explain("Oil prices rose .").explanation.shape == (4,)
