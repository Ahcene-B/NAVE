import warnings

import numpy as np
import pytest

from nave_xai import NAVEImage, ImageExplanation, NaveMemoryWarning
from nave_xai._engine import normalised_entropy


def _toy_class(rgb):                         # a toy "model": dominant colour -> class
    m = rgb.reshape(-1, 3).mean(0)
    return int(np.argmax([m[2] - m[0], m[0] - m[2], 80 - m.mean()]))


@pytest.fixture(scope="module")
def crops():
    from matplotlib import cbook
    from matplotlib.image import imread
    img = imread(cbook.get_sample_data("grace_hopper.jpg", asfileobj=False))
    rng = np.random.default_rng(0)
    return [np.ascontiguousarray(img[y:y + 128, x:x + 128]) for y, x in zip(rng.integers(0, 470, 16), rng.integers(0, 380, 16))]


# -- ImageExplanation-----------------------------------------------------------
def test_explanation_methods():
    e = ImageExplanation(np.array([[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 2, 2]], np.int32), K=3)
    assert e.segmentation().max() + 1 == 3
    b = e.border(1)
    assert set(np.unique(b)) <= {0, 1} and b.dtype == np.int32
    assert b[0, 1] == 1 and b[0, 2] == 1 and b[0, 0] == 0          # inner margins on both sides
    assert np.bincount(e.explanation.ravel(), minlength=e.K).sum() == 12


# -- explain without fit: every call fits afresh ----------------------------------
def test_explain_refits_each_call(small):
    ex = NAVEImage("resnet18", "random", k=5)
    e1 = ex.explain(small)
    assert isinstance(e1, ImageExplanation) and e1.explanation.shape == small.shape[:2] and e1.K == 5
    assert not ex.fitted
    p1 = ex.prototypes.copy()
    ex.explain(small[::-1].copy())
    assert not np.allclose(p1, ex.prototypes)                        # fitted afresh on the new image
    assert ex.palette.shape == (5, 3) and ex.patches.shape == (5, 2, 32, 32, 3)
    assert ex.segment(small).shape == small.shape[:2]


def test_default_model_is_vitt16(small):
    ex = NAVEImage(training="random", k=4)
    exp = ex.explain(small, uncertainty=True)
    assert ex.backbone.name == "vitt16" and ex.layers_ == ["b12"]
    assert 0 <= exp.uncertainty.min() and exp.uncertainty.max() <= 1


# -- fit and predict ------------------------------------------------------------------
def test_fit_predict(crops):
    ex = NAVEImage("resnet18", "random", k=6, classifier=_toy_class).fit(crops)
    assert ex.fitted and ex.prototypes.shape[0] == 6
    assert list(ex.classes) and ex.correlation.shape == (6, len(ex.classes))
    p = ex.prototypes.copy()
    exps = ex.predict(crops[:3], uncertainty=True)
    assert isinstance(exps, list) and len(exps) == 3 and exps[0].uncertainty is not None
    assert np.array_equal(p, ex.prototypes)                          # predict never refits
    assert isinstance(ex.explain(crops[0]), ImageExplanation)             # fitted: explain = predict
    assert np.array_equal(ex.explain(crops[0]).explanation, ex.predict(crops[0]).explanation)


def test_fit_on_all_and_sample(crops):
    a = NAVEImage("resnet18", "random", k=4, fit_on="all").fit(crops[:4])
    s = NAVEImage("resnet18", "random", k=4, fit_on="sample", sample_size=300).fit(crops[:4])
    assert a.prototypes.shape == s.prototypes.shape


def test_streaming_fallback_warns(crops, monkeypatch):
    from nave_xai import memory
    monkeypatch.setattr(memory, "available", lambda: 2 ** 20)        # 1 MB free: too little
    with pytest.warns(UserWarning, match="mini-batch"):
        NAVEImage("resnet18", "random", k=4).fit(crops[:4])
    with pytest.warns(UserWarning, match="Gibbs"):
        NAVEImage("resnet18", "random", k=8, projector="bayesian", bayes_iter=20, bayes_burn_in=10,
             sample_size=500).fit(crops[:3])


def test_groups(crops):
    groups = [0] * 8 + [1] * 8
    ex = NAVEImage("resnet18", "random", k=3).fit(crops, groups=groups, merge_groups=False)
    assert len(ex.palette) == 6 and list(ex.classes) == [0, 1]
    assert ex.correlation.shape == (6, 2)
    g = ex._concept_group
    assert np.all(np.diff(g) >= 0)                                   # ordered by group first
    only = ex.predict(crops[:2], groups=[1, 1])
    assert set(np.unique(np.concatenate([e.explanation.ravel() for e in only]))) <= set(np.nonzero(g == 1)[0])
    merged = NAVEImage("resnet18", "random", k=3).fit(crops, groups=groups)
    with pytest.warns(UserWarning, match="merged"):
        merged.predict(crops[0], groups=[1])


# -- ordering and colours ----------------------------------------------------------------
@pytest.mark.parametrize("method", ["ward", "average", "single"])
def test_reorder_keeps_concepts(crops, method):
    ex = NAVEImage("resnet18", "random", k=5).fit(crops[:6], reorder=None)
    before, palette, protos = ex.predict(crops[0]).explanation, ex.palette.copy(), ex.prototypes.copy()
    order = np.argsort(np.arange(5)[::-1])
    ex.reorder(list(order[::-1]))                                    # a custom order
    custom = np.array(list(order[::-1]))
    assert np.array_equal(ex.prototypes, protos[custom])
    assert np.array_equal(ex.palette, palette[custom])               # colours follow their concepts
    after = ex.predict(crops[0]).explanation
    assert np.array_equal(custom[after], before)
    ex.reorder(method)
    assert ex.prototypes.shape == protos.shape


def test_reorder_by_classes_and_palette(crops):
    ex = NAVEImage("resnet18", "random", k=6, classifier=_toy_class).fit(crops)
    ex.reorder(by="classes", recolor=True, color_vary=["lightness", "hue"], color_spread=0.7)
    assert ex.palette.shape == (6, 3)
    ex.set_palette("viridis")
    ex.set_palette(np.zeros((6, 3), np.uint8))
    with pytest.raises(ValueError):
        ex.set_palette(np.zeros((5, 3), np.uint8))
    with pytest.raises(ValueError):
        NAVEImage("resnet18", "random", k=4).fit(crops[:2]).reorder(by="classes")   # no classes


def test_update_patches(crops):
    ex = NAVEImage("resnet18", "random", k=4).fit(crops[:4])
    with pytest.warns(UserWarning, match="min_size"):
        ex.update_patches(crops[:2], min_size=10_000)                # nothing that large: largest used
    assert ex.patches.shape == (4, 2, 32, 32, 3)
    assert np.array_equal(ex.patches[:, 0], ex.patches[:, 1])


# -- transforms, projectors, saliency, summary ------------------------------------------------------
def test_transforms(small):
    from nave_xai._images import resize
    half = NAVEImage("resnet18", "random", k=4, transforms=lambda im: resize(im, im.shape[0] // 2, im.shape[1] // 2))
    exp = half.explain(small)
    assert exp.explanation.shape == small.shape[:2]                  # back at the original size
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    steps = [lambda im: im.astype(np.float32) / 255, lambda x: (x - mean) / std]
    assert NAVEImage("resnet18", "random", k=4, transforms=steps).explain(small).explanation.shape == small.shape[:2]


def test_projectors(small):
    assert NAVEImage("resnet18", "random", k=4, projector="pca").explain(small).K == 4
    sk = pytest.importorskip("sklearn")
    from sklearn.decomposition import PCA
    from sklearn.mixture import GaussianMixture
    assert NAVEImage("resnet18", "random", projector=GaussianMixture(3, random_state=0)).explain(small).K == 3
    assert NAVEImage("resnet18", "random", projector=PCA(3)).explain(small).K == 3


def test_saliency_and_summarize(crops, rng):
    import matplotlib
    matplotlib.use("Agg")
    ex = NAVEImage("resnet18", "random", k=4, classifier=rng.normal(size=(3, 512))).fit(crops)
    s = ex.saliency(crops[0], target=1)
    assert s.shape == crops[0].shape[:2] and s.dtype == np.float32
    assert len(ex.saliency(crops[:2], target=0)) == 2
    fig = ex.summarize(crops[0], uncertainty=True, saliency=True, patches="largest", target=1)
    assert len(fig.axes) >= 4


def test_memory_warning(small, monkeypatch):
    from nave_xai import memory
    monkeypatch.setattr(memory, "available", lambda: 2 ** 20)
    with pytest.warns(NaveMemoryWarning):
        NAVEImage("resnet18", "random", k=3, fit_on="sample").explain(small)


def test_uncertainty_formula(rng):
    """1 - H(softmax(-d^2 / T)) / log K is the certainty."""
    sq = rng.random((200, 7)) * 5
    for T in (0.1, 1.0, 10.0):
        z = -sq / T
        logp = z - z.max(1, keepdims=True)
        logp -= np.log(np.exp(logp).sum(1, keepdims=True))
        certainty = np.clip(1 - (-(np.exp(logp) * logp).sum(1)) / np.log(7), 0, 1)
        assert np.allclose(1 - normalised_entropy(-sq, T), certainty, atol=1e-6)


# -- windows ------------------------------------------------------------------------------
def test_window_rule():
    from nave_xai import models
    from nave_xai._engine import Tiling
    resnet, vit = models.get_backbone("resnet18", "random"), models.get_backbone("vitt16", "random")
    assert Tiling.plan(600, 512, "auto", resnet).n == 1                    # longer side < 3 x 224: whole
    t = Tiling.plan(700, 500, "auto", resnet)                               # >= 672 px: windows
    assert (t.tile_h, t.margin, t.step_h) == (448, 96, 256)                # 2 x input, ~W/4 margins
    t = Tiling.plan(1000, 900, "auto", vit)
    assert (t.tile_h, t.margin, t.step_h) == (448, 112, 224)               # exactly every W/2
    assert Tiling.plan(700, 500, None, resnet).n == 1
    assert Tiling.plan(2000, 700, "auto", resnet, factor=10).n == 1        # window_factor


def test_windowed_image(rng):
    big = (rng.random((760, 700, 3)) * 255).astype(np.uint8)
    ex = NAVEImage("resnet18", "random", k=4, classifier=rng.normal(size=(3, 512)), fit_on="sample",
                   sample_size=2000).fit([big])
    assert ex._tiling(760, 700).n > 1
    exp = ex.predict(big, uncertainty=True)
    assert exp.explanation.shape == (760, 700) and exp.uncertainty.shape == (760, 700)
    assert ex.classes is not None                                          # windows' logits, weighted
    s = ex.saliency((rng.random((1300, 1100, 3)) * 255).astype(np.uint8), target=0)
    assert s.shape == (1300, 1100)                                         # run at <= 1024 px, resized back
