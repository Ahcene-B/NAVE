import numpy as np
import pytest

from nave_xai import composite
from nave_xai.clustering import BayesianKMeans, ForeignClusterer, KMeans
from nave_xai.composite import CompositeFeatures
from nave_xai.interp import resample


@pytest.mark.parametrize("mode", ["nearest", "linear", "cubic"])
def test_composite_matches_algorithm_1(mode, rng, monkeypatch):
    maps = [rng.normal(size=(8, 12, 10)).astype(np.float32), rng.normal(size=(16, 6, 5)).astype(np.float32)]
    monkeypatch.setattr(composite, "CHUNK_FLOATS", 50)        # force many chunks
    comp = CompositeFeatures(maps, mode)
    got = np.concatenate([x for _, _, x in comp.chunks()])
    parts = []
    for m in maps:
        u = m.transpose(1, 2, 0)
        if u.shape[:2] != (12, 10):
            u = resample(u, 12, 10, mode)
        u = u / (np.linalg.norm(u, axis=2, keepdims=True) + 1e-8)
        parts.append(u * 8 / (1 + m.shape[0]))
    assert np.allclose(got, np.concatenate(parts, 2), atol=1e-5)


def test_composite_options(rng):
    maps = [rng.normal(size=(4, 6, 6)).astype(np.float32)]
    raw = CompositeFeatures(maps, normalize=False, scale=False).full()
    assert np.allclose(raw, maps[0].transpose(1, 2, 0))
    weighted = CompositeFeatures(maps, normalize=False, scale=False, weights=[2.0]).full()
    assert np.allclose(weighted, 2 * raw)


def _blobs(rng, k, d, n=300, spread=6.0):
    centers = rng.normal(0, spread, (k, d))
    X = np.concatenate([c + rng.normal(0, 1, (n, d)) for c in centers]).astype(np.float32)
    return X, np.repeat(np.arange(k), n)


def _purity(pred, truth):
    return sum(np.bincount(truth[pred == c]).max() for c in np.unique(pred)) / len(truth)


def test_kmeans_and_minibatch(rng):
    X, y = _blobs(rng, 4, 8)
    km = KMeans(4, seed=0).fit(X)
    assert _purity(km.predict(X), y) > 0.99
    mb = KMeans(4, seed=0)
    for chunk in np.array_split(X[rng.permutation(len(X))], 10):
        mb.partial_fit(chunk)
    assert _purity(mb.predict(X), y) > 0.95


def test_bayesian_learns_number_of_clusters(rng):
    X, y = _blobs(rng, 6, 64, n=200)
    b = BayesianKMeans(k=60, iterations=80, burn_in=40, seed=1).fit(X)
    assert b.k_used_ == 6
    assert abs(b.sigma2_ - 1.0) < 0.1
    assert _purity(b.predict(X), y) > 0.99


def test_foreign_clusterers(rng):
    sklearn = pytest.importorskip("sklearn")
    from sklearn.cluster import AgglomerativeClustering, KMeans as SKMeans
    from sklearn.mixture import GaussianMixture
    X, y = _blobs(rng, 3, 5)
    for obj, kind in [(SKMeans(3, n_init=2, random_state=0), "transform"),
                      (GaussianMixture(3, random_state=0), "proba"),
                      (AgglomerativeClustering(3), "means")]:
        f = ForeignClusterer(obj).fit(X)
        assert f.kind == kind
        assert _purity(f.scores(X).argmax(1), y) > 0.99
