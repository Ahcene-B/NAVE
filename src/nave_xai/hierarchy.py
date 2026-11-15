# SPDX-License-Identifier: MIT
"""Ordering concepts with a dendrogram of their prototypes.

After k-means, concept ids are arbitrary. Agglomerative clustering of the prototypes gives a
tree; renumbering the concepts in its leaf order puts similar concepts next to each other, so
an ordered palette gives them similar colours.

* "ward": merges the pair of concepts that least increases the within-concept sum of squares.
  Computed from prototypes and concept sizes, this is exact: the Ward cost of merging two
  clusters, n_a n_b / (n_a + n_b) |c_a - c_b|^2, depends only on their centroids and sizes.
* "average": average distance between the members of two groups (UPGMA), with prototypes
  weighted by size. An approximation of average linkage over the underlying pixels.
* "single": distance between the two closest prototypes of two groups.

The linkage matrix follows scipy's format, so it can be drawn with scipy's dendrogram.
"""
import numpy as np

METHODS = ("ward", "average", "single")


def linkage(prototypes, sizes=None, method="ward"):
    """Agglomerative clustering of prototypes.

    Parameters
    ----------
    prototypes : (K, D) array
    sizes : (K,) concept sizes (pixel counts or weights); None = equal sizes
    method : "ward" or "average"

    Returns
    -------
    (K - 1, 4) linkage matrix in scipy's format: merged ids, height, number of prototypes.
    Ward heights are sqrt(2 * cost), so they equal Euclidean distances between single points,
    as in scipy.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}.")
    X = np.asarray(prototypes, np.float64)
    K = len(X)
    n = np.ones(K) if sizes is None else np.maximum(np.asarray(sizes, np.float64), 1e-9)
    if K < 2:
        return np.zeros((0, 4))
    sq = (X * X).sum(1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * X @ X.T, 0.0)
    if method == "ward":
        D = n[:, None] * n[None, :] / (n[:, None] + n[None, :]) * d2     # merge costs
    else:
        D = np.sqrt(d2)
    np.fill_diagonal(D, np.inf)

    ids = list(range(K))                     # scipy id of the cluster held in each row
    count = np.ones(K)                       # number of prototypes per cluster
    centroid, weight = X.copy(), n.copy()
    active = np.ones(K, bool)
    Z = np.zeros((K - 1, 4))
    for step in range(K - 1):
        Dm = np.where(active[:, None] & active[None, :], D, np.inf)
        i, j = np.unravel_index(np.argmin(Dm), Dm.shape)
        if ids[i] > ids[j]:
            i, j = j, i
        height = np.sqrt(2.0 * D[i, j]) if method == "ward" else D[i, j]
        Z[step] = [ids[i], ids[j], height, count[i] + count[j]]
        wi, wj = weight[i], weight[j]
        if method == "ward":                 # recompute merge costs from the new centroid
            centroid[i] = (wi * centroid[i] + wj * centroid[j]) / (wi + wj)
            weight[i] = wi + wj
            diff = centroid - centroid[i]
            D[i] = D[:, i] = weight * weight[i] / (weight + weight[i]) * (diff * diff).sum(1)
        elif method == "average":            # Lance-Williams update for weighted average linkage
            D[i] = D[:, i] = (wi * D[i] + wj * D[j]) / (wi + wj)
            weight[i] = wi + wj
        else:                                # single linkage: the closer of the two
            D[i] = D[:, i] = np.minimum(D[i], D[j])
            weight[i] = wi + wj
        D[i, i] = np.inf
        active[j] = False
        count[i] += count[j]
        ids[i] = K + step
    return Z


def leaf_order(Z):
    """Leaves of the tree from left to right (as scipy.cluster.hierarchy.leaves_list)."""
    K = len(Z) + 1
    if K == 1:
        return np.array([0])
    children = {K + s: (int(Z[s, 0]), int(Z[s, 1])) for s in range(len(Z))}
    order, stack = [], [2 * K - 2]
    while stack:
        node = stack.pop()
        if node < K:
            order.append(node)
        else:
            left, right = children[node]
            stack += [right, left]
    return np.array(order)


def concept_order(prototypes, sizes=None, method="ward"):
    """New order of the concepts: position p holds the old concept id order[p]."""
    Z = linkage(prototypes, sizes, method)
    return leaf_order(Z), Z
