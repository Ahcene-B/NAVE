# SPDX-License-Identifier: MIT
"""Projectors: how composite pixels become concepts.

Every projector provides `fit(X)` and `scores(X)`; scores are per-concept values where
higher means "belongs more": minus the squared distance to a prototype for k-means, the
absolute projection on a component for PCA. Scores are what gets interpolated to pixel
resolution, so concept boundaries can be smooth. Third-party objects (scikit-learn style)
are wrapped by `as_clusterer`.
"""
import copy

import numpy as np


def sq_dists(X, C):
    """Squared Euclidean distances between rows of X (n, d) and C (k, d)."""
    d = (X * X).sum(1)[:, None] - 2.0 * (X @ C.T) + (C * C).sum(1)[None, :]
    return np.maximum(d, 0.0)


def kmeans_plus_plus(X, k, rng):
    """k-means++ seeding."""
    n = len(X)
    k = min(k, n)
    C = np.empty((k, X.shape[1]), np.float32)
    C[0] = X[rng.integers(n)]
    d2 = sq_dists(X, C[:1])[:, 0]
    for i in range(1, k):
        total = d2.sum()
        C[i] = X[rng.choice(n, p=d2 / total) if total > 0 else rng.integers(n)]
        d2 = np.minimum(d2, sq_dists(X, C[i:i + 1])[:, 0])
    return C


class KMeans:
    """k-means: k-means++ seeding, Lloyd iterations, and mini-batch updates (Sculley 2010).

    Parameters
    ----------
    k : number of clusters
    iterations : maximum Lloyd iterations
    seed : random seed
    """

    def __init__(self, k=10, iterations=50, seed=0):
        self.k, self.iterations, self.seed = k, iterations, seed
        self.rng = np.random.default_rng(seed)
        self.centroids_ = None
        self.counts_ = None
        self.inertia_ = None

    def fit(self, X, init=None):
        X = np.asarray(X, np.float32)
        C = kmeans_plus_plus(X, self.k, self.rng) if init is None or len(init) > len(X) \
            else np.array(init, np.float32)
        for _ in range(self.iterations):
            D = sq_dists(X, C)
            labels = D.argmin(1)
            new = C.copy()
            for j in range(len(C)):
                members = labels == j
                new[j] = X[members].mean(0) if members.any() else X[D.min(1).argmax()]
            done = np.abs(new - C).max() < 1e-5
            C = new
            if done:
                break
        D = sq_dists(X, C)
        self.centroids_ = C
        self.counts_ = np.bincount(D.argmin(1), minlength=len(C)).astype(np.float64)
        self.inertia_ = float(D.min(1).sum())
        return self

    def partial_fit(self, X):
        """One mini-batch update with the rows of X (seeds itself on the first call)."""
        X = np.asarray(X, np.float32)
        if self.centroids_ is None:
            self.centroids_ = kmeans_plus_plus(X, self.k, self.rng)
            self.counts_ = np.zeros(len(self.centroids_))
        labels = sq_dists(X, self.centroids_).argmin(1)
        for j in np.unique(labels):
            members = X[labels == j]
            self.counts_[j] += len(members)
            step = len(members) / self.counts_[j]
            self.centroids_[j] += step * (members.mean(0) - self.centroids_[j])
        return self

    def scores(self, X):
        return -sq_dists(np.asarray(X, np.float32), self.centroids_)

    def predict(self, X):
        return self.scores(X).argmax(1)

    def transform(self, X):
        return np.sqrt(-self.scores(X))

    @property
    def weights_(self):
        return None if self.counts_ is None else self.counts_ / max(self.counts_.sum(), 1)


class BayesianKMeans:
    """k-means with a Dirichlet prior on cluster weights, fitted by Gibbs sampling.

    A finite approximation of a Dirichlet-process mixture: K is a large upper bound and
    the sparse prior Dirichlet(alpha / K) leaves unneeded clusters empty, so the number
    of clusters actually used is learned from the data. Model:

        pi ~ Dirichlet(alpha/K, ..., alpha/K)        z_i ~ Categorical(pi)
        mu_k ~ N(m0, s0^2 I)                         x_i ~ N(mu_{z_i}, sigma^2 I)
        sigma^2 ~ InvGamma(a0, b0)                   (one shared isotropic variance)

    Sampling: assignments are drawn from their predictive probabilities with the cluster
    means and weights integrated out, leaving each point out of its own cluster; then
    sigma^2 is drawn given sampled means. Starting from K k-means clusters, duplicate
    clusters merge and empty ones stay empty. After burn-in, the posterior means of the
    centroids and weights are averaged (aligned against label switching); pixels are then
    assigned with these posterior mean parameters.

    In high dimensions the model also explains rare points (region boundaries, noise) with
    many tiny clusters. Clusters whose posterior weight is below `min_weight` are dropped
    after sampling and the rest renormalised (min_weight=0 keeps them all).

    Parameters
    ----------
    k : upper bound on the number of clusters
    alpha : Dirichlet concentration (larger: more clusters a priori)
    iterations, burn_in : Gibbs sweeps in total, and discarded at the start
    min_weight : drop clusters with less posterior weight than this
    seed : random seed
    progress : optional callable(fraction, text)
    """

    def __init__(self, k=100, alpha=1.0, iterations=200, burn_in=100, min_weight=0.005, seed=0, progress=None):
        if burn_in >= iterations:
            raise ValueError("burn_in must be smaller than iterations.")
        self.k, self.alpha, self.iterations, self.burn_in = k, alpha, iterations, burn_in
        self.min_weight, self.progress = min_weight, progress
        self.rng = np.random.default_rng(seed)
        self.centroids_ = self.weights_ = self.sigma2_ = None
        self.k_used_ = None

    @staticmethod
    def _cluster_sums(X, z, K):
        order = np.argsort(z, kind="stable")
        counts = np.bincount(z, minlength=K)
        sums = np.zeros((K, X.shape[1]), np.float64)
        used = np.nonzero(counts)[0]
        starts = np.r_[0, np.cumsum(counts[used])[:-1]]
        sums[used] = np.add.reduceat(X[order], starts, axis=0)
        return sums, counts.astype(np.float64)

    def fit(self, X):
        rng = self.rng
        X = np.asarray(X, np.float32)
        n, d = X.shape
        K = min(self.k, n)
        a = self.alpha / K
        m0 = X.mean(0).astype(np.float64)
        v = float(X.var(0).mean()) + 1e-12            # prior variance of centroids (empirical Bayes)
        a0, b0 = 1.0, v
        x2_total = float(np.einsum("ij,ij->", X, X, dtype=np.float64))
        rows = np.arange(n)

        z = KMeans(K, iterations=10, seed=int(rng.integers(2 ** 31))).fit(X).predict(X)
        sums, counts = self._cluster_sums(X, z, K)
        sigma2 = max(float(x2_total - (sums ** 2).sum(1)[counts > 0].__truediv__(counts[counts > 0]).sum())
                     / (n * d), 1e-12)

        ref = None
        sum_m, count_m = np.zeros((K, d)), np.zeros(K)
        sum_w, sum_sigma2, kept = np.zeros(K), 0.0, 0
        xx = np.einsum("ij,ij->i", X, X, dtype=np.float64)          # |x|^2
        xm0 = X.astype(np.float64) @ m0                                # x . m0
        m0m0 = float(m0 @ m0)
        for it in range(self.iterations):
            # z_i ~ p(z_i | z_-i, sigma2): Gaussian predictive with the cluster mean integrated
            # out. Everything follows from G = X S^T (S = per-cluster sums), via
            # m_k = (S_k / sigma2 + m0 / v) / prec_k and |x - m_k|^2 = |x|^2 - 2 x.m_k + |m_k|^2.
            G = (X @ sums.T.astype(np.float32)).astype(np.float64)    # (n, K)
            SS = np.einsum("ij,ij->i", sums, sums)                    # |S_k|^2
            Sm0 = sums @ m0                                           # S_k . m0
            prec = counts / sigma2 + 1.0 / v
            xm = (G / sigma2 + xm0[:, None] / v) / prec[None]
            mm = (SS / sigma2 ** 2 + 2.0 * Sm0 / (sigma2 * v) + m0m0 / v ** 2) / prec ** 2
            var = sigma2 + 1.0 / prec
            logp = (np.log(counts + a)[None] - 0.5 * d * np.log(var)[None]
                    - np.maximum(xx[:, None] - 2.0 * xm + mm[None], 0.0) / (2.0 * var[None]))
            # own cluster without the point: S_own = S_z - x
            g = G[rows, z]
            c_own = counts[z] - 1.0
            prec_own = c_own / sigma2 + 1.0 / v
            xm_own = ((g - xx) / sigma2 + xm0 / v) / prec_own
            SS_own = SS[z] - 2.0 * g + xx
            Sm0_own = Sm0[z] - xm0
            mm_own = (SS_own / sigma2 ** 2 + 2.0 * Sm0_own / (sigma2 * v) + m0m0 / v ** 2) / prec_own ** 2
            var_own = sigma2 + 1.0 / prec_own
            d_own = np.maximum(xx - 2.0 * xm_own + mm_own, 0.0)
            logp[rows, z] = np.log(c_own + a) - 0.5 * d * np.log(var_own) - d_own / (2.0 * var_own)
            z = (logp + rng.gumbel(size=logp.shape)).argmax(1)
            del logp, G, xm
            sums, counts = self._cluster_sums(X, z, K)

            # sigma2 ~ p(sigma2 | x, z, mu), with mu drawn from its posterior
            prec = counts / sigma2 + 1.0 / v
            m = (sums / sigma2 + m0 / v) / prec[:, None]
            mu = m + rng.standard_normal((K, d)) / np.sqrt(prec)[:, None]
            residual = x2_total - 2.0 * float((sums * mu).sum()) + float((counts * (mu * mu).sum(1)).sum())
            sigma2 = 1.0 / rng.gamma(a0 + n * d / 2.0, 1.0 / (b0 + max(residual, 0.0) / 2.0))

            if self.progress and it % 10 == 0:
                self.progress(it / self.iterations, f"Gibbs sweep {it}/{self.iterations}")
            if it < self.burn_in:
                continue
            m = (sums / sigma2 + m0 / v) / (counts / sigma2 + 1.0 / v)[:, None]
            if ref is None:
                ref = m.copy()
            perm = _align(m.astype(np.float32), ref.astype(np.float32))
            used = counts > 0
            sum_m[perm[used]] += m[used]
            count_m[perm[used]] += 1
            sum_w[perm] += (counts + a) / (n + self.alpha)
            sum_sigma2 += sigma2
            kept += 1

        empty = count_m == 0
        centroids = np.where(empty[:, None], ref, sum_m / np.maximum(count_m, 1)[:, None])
        weights = sum_w / kept
        keep = weights >= self.min_weight
        if not keep.any():
            keep = weights == weights.max()
        self.centroids_ = centroids[keep].astype(np.float32)
        self.weights_ = weights[keep] / weights[keep].sum()
        self.sigma2_ = sum_sigma2 / kept
        self.k_used_ = len(np.unique(self.scores(X).argmax(1)))
        return self

    def scores(self, X):
        X = np.asarray(X, np.float32)
        return (np.log(np.maximum(self.weights_, 1e-300))[None]
                - sq_dists(X, self.centroids_) / (2.0 * self.sigma2_)).astype(np.float32)

    def predict(self, X):
        return self.scores(X).argmax(1)


class PCA:
    """Principal components as concepts (as the DINO visualisations and the paper's Figure 2):
    a pixel goes to the component with the largest absolute projection.

    Fitted from the covariance of the composite pixels, so it can be accumulated over chunks
    (`partial_fit`) and is exact on all pixels."""

    def __init__(self, k=10):
        self.k = k
        self._n, self._sum, self._outer = 0, None, None
        self.components_ = self.mean_ = self.explained_variance_ = None

    def partial_fit(self, X):
        X = np.asarray(X, np.float64)
        if self._sum is None:
            self._sum, self._outer = np.zeros(X.shape[1]), np.zeros((X.shape[1], X.shape[1]))
        self._n += len(X)
        self._sum += X.sum(0)
        self._outer += X.T @ X
        self._finish()
        return self

    def fit(self, X):
        self._n, self._sum, self._outer = 0, None, None
        return self.partial_fit(X)

    def _finish(self):
        mean = self._sum / self._n
        cov = self._outer / self._n - np.outer(mean, mean)
        values, vectors = np.linalg.eigh(cov)
        top = np.argsort(values)[::-1][:self.k]
        self.mean_ = mean.astype(np.float32)
        self.components_ = vectors[:, top].T.astype(np.float32)
        self.explained_variance_ = values[top]

    def scores(self, X):
        return np.abs((np.asarray(X, np.float32) - self.mean_) @ self.components_.T)

    def predict(self, X):
        return self.scores(X).argmax(1)

    @property
    def centroids_(self):
        return self.components_


def _align(mu, ref):
    """Greedy matching of the rows of `mu` to the rows of `ref` by distance."""
    D = sq_dists(mu, ref)
    K = len(mu)
    perm = -np.ones(K, int)
    taken = np.zeros(K, bool)
    for flat in np.argsort(D, axis=None):
        i, j = divmod(int(flat), K)
        if perm[i] < 0 and not taken[j]:
            perm[i], taken[j] = j, True
    return perm


class ForeignClusterer:
    """Wraps a scikit-learn-style clusterer so NAVE can use it.

    How concept scores are obtained, in order of preference:
      transform(X) and predict(X)   a clusterer (k-means): transform gives distances,
                                    score = -distance^2
      predict_proba(X)              score = log probability (Gaussian mixtures)
      transform(X) only             a projection (PCA, NMF, ICA): score = |projection|
      predict(X)                    one-hot scores (no smooth boundaries, no uncertainty)
      fit_predict(X) only           cluster means are computed from the fit, then used as k-means
    """

    def __init__(self, obj):
        self.obj = obj
        if hasattr(obj, "transform") and hasattr(obj, "predict"):
            self.kind = "transform"
        elif hasattr(obj, "predict_proba"):
            self.kind = "proba"
        elif hasattr(obj, "transform"):
            self.kind = "projection"
        elif hasattr(obj, "predict_proba"):
            self.kind = "proba"
        elif hasattr(obj, "predict"):
            self.kind = "predict"
        elif hasattr(obj, "fit_predict"):
            self.kind = "means"
        else:
            raise TypeError(f"{type(obj).__name__} has none of transform, predict_proba, predict, fit_predict.")
        self.prefit = _is_fitted(obj)
        self._means = None
        self._k = None

    @property
    def has_soft_scores(self):
        return self.kind != "predict"

    @property
    def supports_partial_fit(self):
        return hasattr(self.obj, "partial_fit")

    def fit(self, X):
        if self.kind == "means":
            labels = np.asarray(self.obj.fit_predict(X))
            ids = np.unique(labels)
            self._means = np.stack([X[labels == i].mean(0) for i in ids]).astype(np.float32)
        else:
            self.obj.fit(X)
        self.prefit = True
        return self

    def partial_fit(self, X):
        self.obj.partial_fit(X)
        self.prefit = True
        return self

    def scores(self, X):
        if self.kind == "means":
            return -sq_dists(np.asarray(X, np.float32), self._means)
        if self.kind == "transform":
            return -np.square(np.asarray(self.obj.transform(X), np.float32))
        if self.kind == "projection":
            return np.abs(np.asarray(self.obj.transform(X), np.float32))
        if self.kind == "proba":
            return np.log(np.maximum(np.asarray(self.obj.predict_proba(X), np.float32), 1e-12))
        labels = np.asarray(self.obj.predict(X))
        if self._k is None:
            self._k = int(getattr(self.obj, "n_clusters", getattr(self.obj, "n_components", labels.max() + 1)))
        out = np.zeros((len(labels), max(self._k, int(labels.max()) + 1)), np.float32)
        out[np.arange(len(labels)), labels] = 1.0
        return out

    @property
    def centroids_(self):
        for name in ("cluster_centers_", "means_", "components_"):
            if hasattr(self.obj, name):
                return np.asarray(getattr(self.obj, name), np.float32)
        return self._means

    def fresh(self):
        """An unfitted copy, for fitting one clusterer per window."""
        try:
            from sklearn.base import clone
            return ForeignClusterer(clone(self.obj))
        except Exception:
            return ForeignClusterer(copy.deepcopy(self.obj))


def _is_fitted(obj):
    """scikit-learn convention: fitted estimators have attributes ending in '_'."""
    return any(k.endswith("_") and not k.startswith("_") for k in vars(obj)) if hasattr(obj, "__dict__") else False


def as_clusterer(spec, k, seed, alpha=1.0, iterations=200, burn_in=100):
    """Build the projector for `projector=` arguments: None/"kmeans", "bayesian", "pca", or an object."""
    if spec is None or (isinstance(spec, str) and spec == "kmeans"):
        return KMeans(k, seed=seed)
    if isinstance(spec, str) and spec == "bayesian":
        return BayesianKMeans(k, alpha=alpha, iterations=iterations, burn_in=burn_in, seed=seed)
    if isinstance(spec, str) and spec == "pca":
        return PCA(k)
    if isinstance(spec, str):
        raise ValueError(f"Unknown projector {spec!r}: use 'kmeans', 'bayesian', 'pca' or an object.")
    if isinstance(spec, (KMeans, BayesianKMeans, PCA, ForeignClusterer)):
        return spec
    return ForeignClusterer(spec)


def fresh_copy(clusterer, seed):
    """An unfitted clusterer of the same kind (used per window in single-pass mode)."""
    if isinstance(clusterer, KMeans):
        return KMeans(clusterer.k, clusterer.iterations, seed)
    if isinstance(clusterer, PCA):
        return PCA(clusterer.k)
    if isinstance(clusterer, BayesianKMeans):
        return BayesianKMeans(clusterer.k, clusterer.alpha, clusterer.iterations, clusterer.burn_in,
                              clusterer.min_weight, seed)
    if isinstance(clusterer, ForeignClusterer):
        return clusterer.fresh()
    raise ValueError(f"{type(clusterer).__name__} cannot be fitted per window; use two_pass=True.")


def is_fitted(clusterer):
    if isinstance(clusterer, ForeignClusterer):
        return clusterer.prefit
    return getattr(clusterer, "centroids_", None) is not None
