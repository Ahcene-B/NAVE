#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
NAVE superpixels worker (numpy + onnxruntime).

NAVE (Boubekki et al., arXiv:2411.19700, Algorithm 1) is run on overlapping
448 px windows. Each window is clustered into K clusters (k-means warm-started
from the neighbouring window's centroids, so neighbours split regions alike),
upsampled to full resolution and split into 8-connected segments. Segments of
neighbouring windows are joined when they agree in the 64 px overlap, unless
some window keeps them apart, and the windows' centres are stitched together.

Run by the GIMP plugin as a subprocess:   python nave_core.py job.json
"""
import json
import math
import os
import sys

import numpy as np

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
MARGIN = 32            # px discarded at each window edge; overlap = 2 * MARGIN
N_BLOCKS = {"resnet18": 4, "resnet50": 4, "vits16": 12}
KMEANS_ITERS = 50


# --------------------------------------------------------------------------
# Block selection (paper notation)
# --------------------------------------------------------------------------
def parse_blocks(spec, model):
    """ResNet: 'N234', 'N34', 'N4' (one digit per block).
    ViT: 'N:12', 'N:9-12', '9,12' (block numbers 1-12; '-' separates, as in the paper)."""
    s = spec.strip().upper()
    s = (s[1:] if s.startswith("N") else s).lstrip(":").strip()
    if model.startswith("resnet"):
        blocks = {int(c) for c in s if c.isdigit()}
    else:
        blocks = {int(p) for p in s.replace("-", ",").replace(" ", ",").split(",") if p}
    n = N_BLOCKS[model]
    blocks = sorted(blocks)
    if not blocks or blocks[0] < 1 or blocks[-1] > n:
        raise ValueError(f"Invalid blocks '{spec}' for {model}: use block numbers 1-{n}.")
    return blocks


# --------------------------------------------------------------------------
# Separable interpolation as matrices
# --------------------------------------------------------------------------
def _keys(d, a=-0.75):
    d = np.abs(d)
    return np.where(d <= 1, ((a + 2) * d - (a + 3)) * d * d + 1,
                    np.where(d < 2, ((a * d - 5 * a) * d + 8 * a) * d - 4 * a, 0.0))


def resize_coords(n_in, n_out):
    return (np.arange(n_out) + 0.5) * (n_in / n_out) - 0.5


def interp_matrix(coords, n_in, mode):
    """(len(coords), n_in) matrix sampling a length-n_in signal at fractional `coords`."""
    coords = np.asarray(coords, np.float64)
    rows = np.arange(len(coords))
    A = np.zeros((len(coords), n_in), np.float32)
    if mode == "nearest":
        A[rows, np.clip(np.floor(coords + 0.5).astype(int), 0, n_in - 1)] = 1.0
    elif mode == "linear":
        x = np.clip(coords, 0, n_in - 1)
        i0 = np.floor(x).astype(int)
        t = x - i0
        np.add.at(A, (rows, i0), 1 - t)
        np.add.at(A, (rows, np.minimum(i0 + 1, n_in - 1)), t)
    elif mode == "cubic":
        i0 = np.floor(coords).astype(int)
        t = coords - i0
        for k in (-1, 0, 1, 2):
            np.add.at(A, (rows, np.clip(i0 + k, 0, n_in - 1)), _keys(t - k))
    else:
        raise ValueError(f"Unknown upsampling '{mode}'")
    return A


def apply_separable(Ah, Aw, X):
    """X: (h, w, c) -> (rows of Ah, rows of Aw, c)."""
    h, w, c = X.shape
    Y = (Ah @ X.reshape(h, w * c)).reshape(Ah.shape[0], w, c)
    return np.matmul(Aw, Y)


# --------------------------------------------------------------------------
# 8-connected components (run-length + label propagation, numpy only)
# --------------------------------------------------------------------------
def _resolve(n, u, v):
    """Union of node pairs (u, v); returns the root (smallest id) of every node."""
    lab = np.arange(n)
    while len(u):
        m = np.minimum(lab[u], lab[v])
        np.minimum.at(lab, u, m)
        np.minimum.at(lab, v, m)
        nxt = lab[lab]
        while not np.array_equal(nxt, lab):
            lab, nxt = nxt, nxt[nxt]
        if np.array_equal(lab[u], lab[v]):
            break
    return lab


def connected_components(img):
    """Label 8-connected regions of equal value. Returns int32 labels 0..n-1 and n."""
    H, W = img.shape
    start = np.ones((H, W), bool)
    start[:, 1:] = img[:, 1:] != img[:, :-1]
    rows, starts = np.nonzero(start)
    ends = np.empty_like(starts)
    ends[:-1] = starts[1:]
    last = np.r_[rows[1:] != rows[:-1], True]
    ends[last] = W
    vals = img[rows, starts]
    ptr = np.searchsorted(rows, np.arange(H + 1))
    us, vs = [], []
    for r in range(1, H):
        a0, a1, b1 = ptr[r - 1], ptr[r], ptr[r + 1]
        As, Ae, Bs, Be = starts[a0:a1], ends[a0:a1], starts[a1:b1], ends[a1:b1]
        lo = np.searchsorted(Ae, Bs, "left")          # runs above whose end reaches B's start (diagonal incl.)
        hi = np.searchsorted(As, Be, "right")         # ... and whose start is not past B's end
        cnt = np.maximum(hi - lo, 0)
        tot = int(cnt.sum())
        if not tot:
            continue
        b = np.repeat(np.arange(len(Bs)), cnt)
        a = np.repeat(lo, cnt) + np.arange(tot) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        same = vals[a0 + a] == vals[a1 + b]
        us.append(a0 + a[same])
        vs.append(a1 + b[same])
    u = np.concatenate(us) if us else np.zeros(0, int)
    v = np.concatenate(vs) if vs else np.zeros(0, int)
    root = _resolve(len(starts), u, v)
    _, run_label = np.unique(root, return_inverse=True)
    out = np.repeat(run_label.astype(np.int32), ends - starts).reshape(H, W)
    return out, int(run_label.max()) + 1


# --------------------------------------------------------------------------
# NAVE composite pixels for one window, computed in row chunks (bounded memory)
# --------------------------------------------------------------------------
CHUNK_FLOATS = 16 * 2 ** 20        # ~64 MB of composite pixels at a time
KMEANS_SAMPLE = 20000              # composite pixels used to fit k-means


class WindowFeatures:
    """Algorithm 1, lines 6-11, for one window: selected blocks upsampled to the
    shallowest block's grid, L2-normalized per pixel, scaled by C_min/(1+C_b), concatenated."""

    def __init__(self, feats, blocks, mode):
        maps = [feats[f"b{b}"][0] for b in blocks]
        self.c_ref, self.gh, self.gw = maps[0].shape
        self.parts = []
        for m in maps:
            C, mh, mw = m.shape
            u = np.ascontiguousarray(m.transpose(1, 2, 0), np.float32).reshape(mh, mw * C)
            Ah = None if mh == self.gh else interp_matrix(resize_coords(mh, self.gh), mh, mode)
            Aw = None if mw == self.gw else interp_matrix(resize_coords(mw, self.gw), mw, mode)
            self.parts.append((u, C, mw, Ah, Aw))
        self.dim = sum(p[1] for p in self.parts)
        self.step = max(1, CHUNK_FLOATS // (self.gw * self.dim))

    def rows(self, r0, r1):
        out = []
        for u, C, mw, Ah, Aw in self.parts:
            x = (u[r0:r1] if Ah is None else Ah[r0:r1] @ u).reshape(r1 - r0, mw, C)
            if Aw is not None:
                x = np.matmul(Aw, x)
            x = x / (np.linalg.norm(x, axis=2, keepdims=True) + 1e-8)
            out.append(x * (self.c_ref / (1.0 + C)))
        return np.concatenate(out, axis=2)

    def chunks(self):
        for r0 in range(0, self.gh, self.step):
            r1 = min(self.gh, r0 + self.step)
            yield r0, r1, self.rows(r0, r1)

    def sample(self, n, rng):
        cells = self.gh * self.gw
        idx = np.sort(rng.choice(cells, min(n, cells), replace=False))
        out = []
        for r0, r1, c in self.chunks():
            sel = idx[(idx >= r0 * self.gw) & (idx < r1 * self.gw)] - r0 * self.gw
            out.append(c.reshape(-1, self.dim)[sel])
        return np.concatenate(out)

    def distances(self, C):
        dist = np.empty((self.gh, self.gw, len(C)), np.float32)
        for r0, r1, c in self.chunks():
            dist[r0:r1] = sq_dists(c.reshape(-1, self.dim), C).reshape(r1 - r0, self.gw, -1)
        return dist


def sq_dists(X, C):
    return np.maximum((X * X).sum(1)[:, None] - 2.0 * (X @ C.T) + (C * C).sum(1)[None, :], 0.0)


def kmeans(X, k, rng, init=None, iters=KMEANS_ITERS):
    """Lloyd's k-means. Starts from `init` centroids if given, else k-means++."""
    n = len(X)
    if init is not None and len(init) <= n:
        C = init.astype(np.float32).copy()
    else:
        k = min(k, n)
        C = np.empty((k, X.shape[1]), np.float32)
        C[0] = X[rng.integers(n)]
        d2 = sq_dists(X, C[:1])[:, 0]
        for i in range(1, k):                        # k-means++
            tot = d2.sum()
            C[i] = X[rng.choice(n, p=d2 / tot) if tot > 0 else rng.integers(n)]
            d2 = np.minimum(d2, sq_dists(X, C[i:i + 1])[:, 0])
    k = len(C)
    for _ in range(iters):                           # Lloyd
        D = sq_dists(X, C)
        lab = D.argmin(1)
        new = C.copy()
        for j in range(k):
            sel = lab == j
            new[j] = X[sel].mean(0) if sel.any() else X[D.min(1).argmax()]
        done = np.abs(new - C).max() < 1e-5
        C = new
        if done:
            break
    return C


def pixel_clusters(dist, win, mode, p0, p1):
    """Nearest cluster for window pixels [p0, p1) x [p0, p1), from the grid's
    distance maps upsampled with the chosen interpolation."""
    gh, gw, K = dist.shape
    px = np.arange(p0, p1) + 0.5
    Ah = interp_matrix(px * gh / win - 0.5, gh, mode)
    Aw = interp_matrix(px * gw / win - 0.5, gw, mode)
    flat = dist.reshape(gh, gw * K)
    out = np.empty((p1 - p0, p1 - p0), np.int32)
    step = max(1, (32 * 2 ** 20) // ((p1 - p0) * K))
    for r in range(0, p1 - p0, step):
        t = (Ah[r:r + step] @ flat).reshape(-1, gw, K)
        out[r:r + step] = np.matmul(Aw, t).argmin(2)
    return out


def match_strips(a, b, min_share=0.1):
    """Candidate joins between segment ids a and b covering the same overlap pixels.
    A pair qualifies when it covers at least half of one segment's overlap area and at
    least `min_share` of the other's. Returns (ids_a, ids_b, iou) for ranking."""
    a, b = a.ravel().astype(np.int64), b.ravel().astype(np.int64)
    pairs, inter = np.unique(np.stack([a, b]), axis=1, return_counts=True)
    ua, ca = np.unique(a, return_counts=True)
    ub, cb = np.unique(b, return_counts=True)
    area_a = ca[np.searchsorted(ua, pairs[0])]
    area_b = cb[np.searchsorted(ub, pairs[1])]
    fa, fb = inter / area_a, inter / area_b
    keep = (np.maximum(fa, fb) >= 0.5) & (np.minimum(fa, fb) >= min_share)
    iou = inter / (area_a + area_b - inter)
    return pairs[0, keep], pairs[1, keep], iou[keep]


def touching_pairs(seg):
    """Pairs of different segment ids that touch inside one window (cannot-link)."""
    p = np.concatenate([np.stack([seg[:, :-1].ravel(), seg[:, 1:].ravel()]),
                        np.stack([seg[:-1, :].ravel(), seg[1:, :].ravel()])], axis=1)
    p = p[:, p[0] != p[1]]
    return np.unique(np.sort(p, axis=0), axis=1)


def join_segments(n, a, b, score, apart_a, apart_b):
    """Greedy union of candidate pairs, strongest first. A join is skipped when it would
    merge two segments that touch inside some window (that window keeps them apart)."""
    parent = np.arange(n)
    members, apart = {}, {}
    for x, y in zip(apart_a.tolist(), apart_b.tolist()):
        apart.setdefault(x, set()).add(y)
        apart.setdefault(y, set()).add(x)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in np.argsort(-score, kind="stable"):
        ra, rb = find(a[i]), find(b[i])
        if ra == rb:
            continue
        ma, mb = members.get(ra, {ra}), members.get(rb, {rb})
        xa, xb = apart.get(ra, set()), apart.get(rb, set())
        if len(ma) < len(mb):
            ra, rb, ma, mb, xa, xb = rb, ra, mb, ma, xb, xa
        if not xa.isdisjoint(mb):
            continue
        parent[rb] = ra
        members[ra] = ma | mb
        apart[ra] = xa | xb
        members.pop(rb, None)
        apart.pop(rb, None)
    return np.array([find(x) for x in range(n)])


# --------------------------------------------------------------------------
# Whole image
# --------------------------------------------------------------------------
def windows(rgb, win):
    """Yield (iy, ix, ny, nx, y0, x0, tile) over overlapping windows of the padded image."""
    H, W, _ = rgb.shape
    m = MARGIN
    step = win - 2 * m
    ny, nx = math.ceil(H / step), math.ceil(W / step)
    pad = ((m, ny * step + m - H), (m, nx * step + m - W), (0, 0))
    img = rgb.astype(np.float32) / 255.0
    try:
        img = np.pad(img, pad, mode="reflect")
    except ValueError:
        img = np.pad(img, pad, mode="edge")
    img = (img - MEAN) / STD
    for iy in range(ny):
        for ix in range(nx):
            y0, x0 = iy * step, ix * step
            yield iy, ix, ny, nx, y0, x0, np.ascontiguousarray(
                img[y0:y0 + win, x0:x0 + win].transpose(2, 0, 1)[None])


def segment_single_pass(rgb, backbone, blocks, mode, k, seed, progress):
    """K clusters per window, segments joined across windows. Returns segment labels."""
    H, W, _ = rgb.shape
    win, m = backbone.window, MARGIN
    step = win - 2 * m
    stitched = None
    us, vs, scores, apart = [], [], [], []
    offset = 0
    left_c, top_c, prev_right, prev_bottom = None, {}, None, {}
    for iy, ix, ny, nx, y0, x0, tile in windows(rgb, win):
        if stitched is None:
            stitched = np.empty((ny * step, nx * step), np.int32)
        rng = np.random.default_rng([seed, iy, ix])
        wf = WindowFeatures(backbone(tile), blocks, mode)
        init = left_c if ix > 0 else top_c.get(ix)
        C = kmeans(wf.sample(KMEANS_SAMPLE, rng), k, rng, init)
        left_c, top_c[ix] = C, C
        seg, n = connected_components(pixel_clusters(wf.distances(C), win, mode, 0, win))
        del wf
        gid = seg + offset
        offset += n
        apart.append(touching_pairs(gid))
        for prev, cur in ((prev_right, gid[:, :2 * m]) if ix > 0 else (None, None),
                          (prev_bottom.get(ix), gid[:2 * m, :]) if iy > 0 else (None, None)):
            if prev is not None:
                a, b, sc = match_strips(prev, cur)
                us.append(a); vs.append(b); scores.append(sc)
        prev_right, prev_bottom[ix] = gid[:, win - 2 * m:], gid[win - 2 * m:, :]
        stitched[y0:y0 + step, x0:x0 + step] = gid[m:m + step, m:m + step]
        done = iy * nx + ix + 1
        progress(0.2 + 0.7 * done / (ny * nx), f"NAVE: window {done}/{ny * nx}")

    progress(0.92, "NAVE: joining windows")
    cat = lambda xs, dt: np.concatenate(xs) if xs else np.zeros(0, dt)
    ap = np.concatenate(apart, axis=1) if apart else np.zeros((2, 0), int)
    root = join_segments(offset, cat(us, int), cat(vs, int), cat(scores, float), ap[0], ap[1])
    labels, _ = connected_components(root[stitched[:H, :W]])
    return labels


def segment_two_pass(rgb, backbone, blocks, mode, k, seed, progress):
    """One k-means for the whole image: pass 1 samples composite pixels from every
    window and fits it, pass 2 assigns every pixel. Returns (segment labels, cluster map)."""
    H, W, _ = rgb.shape
    win, m = backbone.window, MARGIN
    step = win - 2 * m
    ny, nx = math.ceil(H / step), math.ceil(W / step)
    per_window = math.ceil(KMEANS_SAMPLE / (ny * nx))
    samples = []
    for iy, ix, _, _, _, _, tile in windows(rgb, win):
        rng = np.random.default_rng([seed, iy, ix])
        samples.append(WindowFeatures(backbone(tile), blocks, mode).sample(per_window, rng))
        done = iy * nx + ix + 1
        progress(0.2 + 0.35 * done / (ny * nx), f"NAVE: pass 1, window {done}/{ny * nx}")
    progress(0.56, "NAVE: clustering")
    C = kmeans(np.concatenate(samples), k, np.random.default_rng(seed))
    del samples

    clusters = np.empty((ny * step, nx * step), np.int32)
    for iy, ix, _, _, y0, x0, tile in windows(rgb, win):
        dist = WindowFeatures(backbone(tile), blocks, mode).distances(C)
        clusters[y0:y0 + step, x0:x0 + step] = pixel_clusters(dist, win, mode, m, m + step)
        done = iy * nx + ix + 1
        progress(0.58 + 0.34 * done / (ny * nx), f"NAVE: pass 2, window {done}/{ny * nx}")
    clusters = np.ascontiguousarray(clusters[:H, :W])
    progress(0.93, "NAVE: segments")
    labels, _ = connected_components(clusters)
    return labels, clusters


def render_borders(labels, color, width=1):
    """Transparent layer with segment borders `width` px wide."""
    H, W = labels.shape
    edge = np.zeros((H, W), bool)
    edge[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    edge[:-1, :] |= labels[:-1, :] != labels[1:, :]
    for i in range(width - 1):                       # grow alternately right/down and left/up
        g = edge.copy()
        if i % 2 == 0:
            g[:, 1:] |= edge[:, :-1]; g[1:, :] |= edge[:-1, :]; g[1:, 1:] |= edge[:-1, :-1]
        else:
            g[:, :-1] |= edge[:, 1:]; g[:-1, :] |= edge[1:, :]; g[:-1, :-1] |= edge[1:, 1:]
        edge = g
    out = np.zeros((H, W, 4), np.uint8)
    out[edge] = color
    return out


def render_average(rgb, labels):
    """Opaque layer, each segment filled with its mean color."""
    H, W = labels.shape
    flat = labels.ravel()
    n = int(flat.max()) + 1
    counts = np.maximum(np.bincount(flat, minlength=n), 1)
    means = np.stack([np.bincount(flat, weights=rgb[..., c].ravel().astype(np.float64), minlength=n)
                      for c in range(3)], 1) / counts[:, None]
    out = np.empty((H, W, 4), np.uint8)
    out[..., :3] = np.round(means[labels]).astype(np.uint8)
    out[..., 3] = 255
    return out


# matplotlib's tab10 then tab20's light tones, like the paper's figures; more are generated
_PALETTE = ["1f77b4", "ff7f0e", "2ca02c", "d62728", "9467bd", "8c564b", "e377c2", "7f7f7f", "bcbd22", "17becf",
            "aec7e8", "ffbb78", "98df8a", "ff9896", "c5b0d5", "c49c94", "f7b6d2", "c7c7c7", "dbdb8d", "9edae5"]


def palette(n):
    import colorsys
    cols = [tuple(int(h[i:i + 2], 16) for i in (0, 2, 4)) for h in _PALETTE]
    i = 0
    while len(cols) < n:                              # golden-ratio hues beyond 20 clusters
        hue = (i * 0.618033988749895) % 1.0
        sat, val = (0.65, 0.9) if i % 2 == 0 else (0.45, 0.7)
        cols.append(tuple(int(round(c * 255)) for c in colorsys.hsv_to_rgb(hue, sat, val)))
        i += 1
    return np.array(cols[:n], np.uint8)


def render_segmentation(clusters, k):
    out = np.empty(clusters.shape + (4,), np.uint8)
    out[..., :3] = palette(k)[clusters]
    out[..., 3] = 255
    return out


def check_window(win):
    win = int(win)
    if win % 32 or not 224 <= win <= 2048:
        raise ValueError(f"Window size must be a multiple of 32 between 224 and 2048 px (got {win}).")
    return win


def run(rgb_bytes, width, height, params, progress=None, backbone=None):
    """R'G'B' u8 bytes in. Returns {name: R'G'B'A u8 bytes} for the layers asked for in
    params["outputs"]: any of "borders", "colors", "segmentation" (two-pass only)."""
    progress = progress or (lambda f, t="": None)
    model = params["model"]
    blocks = parse_blocks(params["blocks"], model)
    seed, k = int(params.get("seed", 0)), int(params["clusters"])
    two_pass = bool(params.get("two_pass", False))
    outputs = list(params.get("outputs", ["borders"]))
    if "segmentation" in outputs and not two_pass:
        raise ValueError("The segmentation layer needs two-pass mode (clusters must be the same in every window).")
    if not outputs:
        raise ValueError("Choose at least one output layer.")
    win = check_window(params.get("window", 448))
    if backbone is None:
        from nave_models import Backbone
        progress(0.0, "NAVE: preparing weights")
        backbone = Backbone(model, params["training"], seed, win, progress)
    rgb = np.frombuffer(rgb_bytes, np.uint8).reshape(height, width, 3)
    if two_pass:
        labels, clusters = segment_two_pass(rgb, backbone, blocks, params["upsample"], k, seed, progress)
    else:
        labels = segment_single_pass(rgb, backbone, blocks, params["upsample"], k, seed, progress)
    progress(0.97, "NAVE: drawing")
    layers = {}
    if "colors" in outputs:
        layers["colors"] = render_average(rgb, labels).tobytes()
    if "segmentation" in outputs:
        layers["segmentation"] = render_segmentation(clusters, k).tobytes()
    if "borders" in outputs:
        color = np.array(params.get("border_color", [0, 0, 0, 255]), np.uint8)
        layers["borders"] = render_borders(labels, color, int(params.get("border_width", 1))).tobytes()
    progress(1.0, "NAVE: done")
    return layers


OUT_OF_MEMORY = ("NAVE ran out of memory. Try a smaller window size, fewer blocks "
                 "(block 1 is the largest), or, with ViT, a much smaller window.")


def main():
    with open(sys.argv[1]) as f:
        job = json.load(f)
    with open(job["input"], "rb") as f:
        data = f.read()

    def progress(frac, text=""):
        print(f"PROGRESS {frac:.4f} {text}", flush=True)

    try:
        layers = run(data, job["width"], job["height"], job["params"], progress)
    except MemoryError:
        sys.exit(OUT_OF_MEMORY)
    except Exception as e:                            # onnxruntime reports allocation failures as errors
        text = str(e).lower()
        if "bad_alloc" in text or "failed to allocate" in text or "unable to allocate" in text:
            sys.exit(OUT_OF_MEMORY)
        raise
    for name, buf in layers.items():
        with open(job["outputs"][name], "wb") as f:
            f.write(buf)


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
