import numpy as np
import pytest

from nave_xai.hierarchy import leaf_order, linkage


@pytest.mark.parametrize("method", ["ward", "average", "single"])
def test_linkage_matches_scipy(method, rng):
    hierarchy = pytest.importorskip("scipy.cluster.hierarchy")
    for _ in range(20):
        X = rng.normal(size=(rng.integers(3, 25), 5))
        Z, S = linkage(X, method=method), hierarchy.linkage(X, method=method)
        assert np.allclose(Z[:, 2], S[:, 2])
        assert np.array_equal(leaf_order(Z), hierarchy.leaves_list(S))


def test_ward_is_exact_from_prototypes(rng):
    A, B = rng.normal(0, 1, (40, 5)), rng.normal(2, 1, (25, 5))
    sse = lambda P: ((P - P.mean(0)) ** 2).sum()
    increase = sse(np.vstack([A, B])) - sse(A) - sse(B)
    Z = linkage(np.stack([A.mean(0), B.mean(0)]), sizes=[40, 25], method="ward")
    assert np.isclose(Z[0, 2] ** 2 / 2, increase)
