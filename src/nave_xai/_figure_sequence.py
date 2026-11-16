# SPDX-License-Identifier: MIT
"""The figures of `NAVESeries.summarize` and `NAVEText.summarize`."""
import numpy as np

from ._layout import FIG_WIDTH, LEFT, RIGHT, columns

CLASS_LIMIT = 30


def _contrast(s):
    """Saliency shown as in the ECG paper: standardized, then a sigmoid."""
    z = (s - s.mean()) / (s.std() + 1e-9)
    return 1 / (1 + np.exp(-z))


def _class_label(names, c):
    try:
        return str(names[c]) if names is not None else str(c)
    except (KeyError, IndexError, TypeError):
        return str(c)


def _correlation_view(explainer):
    R, classes = explainer.correlation, explainer.classes
    if R is None:
        return None, None
    if len(classes) > CLASS_LIMIT:
        R, classes = R[:, :CLASS_LIMIT], classes[:CLASS_LIMIT]
    return R, classes


def _frame(ax, colour):
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(colour)
        sp.set_linewidth(2)


# -- shared pieces ----------------------------------------------------------------------------------
def headline(names, predicted, label):
    """'predicted: X  |  true: Y', with whatever is available (empty if neither)."""
    parts = []
    if predicted is not None:
        parts.append(f"predicted: {_class_label(names, predicted)}")
    if label is not None:
        parts.append(f"true: {_class_label(names, label)}")
    return "   |   ".join(parts)


def saliency_range(spec, values=None):
    """(vmin, vmax) for saliency: a (vmin, vmax) tuple as given, or "auto": centred on 0, up to the
    largest |value|."""
    if isinstance(spec, str):
        if spec != "auto":
            raise ValueError('saliency_range must be "auto" or a (vmin, vmax) tuple.')
        v = np.asarray(values, np.float64) if values is not None else np.zeros(1)
        h = float(np.nanmax(np.abs(v))) if np.isfinite(v).any() else 1.0
        h = max(h, 1e-9)
        return -h, h
    lo, hi = float(spec[0]), float(spec[1])
    if not lo < hi:
        raise ValueError("saliency_range must have vmin < vmax.")
    return lo, hi


def _ticks(lo, hi):
    return [lo, 0.0, hi] if lo < 0 < hi else [lo, hi]


def _concept_matrix(fig, ax, ex, M, row_labels, concepts, n_cols, colorbar, cmap="coolwarm", vmin=-1, vmax=1,
                    ticks=(-1, 0, 1)):
    """A (rows x concepts) matrix, each column headed by the concept's colour and number."""
    from matplotlib import colormaps
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.patches import Rectangle
    from matplotlib.transforms import blended_transform_factory
    n = len(row_labels)
    img = np.full((n, n_cols), np.nan)
    img[:, :len(concepts)] = M[:, concepts]
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest", extent=(0, n_cols, n, 0))
    axes_h = ax.get_position().height * fig.get_figheight()               # inches
    gap, bar = 0.04 / axes_h, 0.09 / axes_h                               # fixed sizes, whatever the row count
    above = blended_transform_factory(ax.transData, ax.transAxes)
    for j, k in enumerate(concepts):
        ax.add_patch(Rectangle((j + 0.04, 1 + gap), 0.92, bar, transform=above,
                               color=np.asarray(ex.palette[k]) / 255, clip_on=False))
    ax.set_xticks(np.arange(len(concepts)) + 0.5, [str(k) for k in concepts], fontsize=8)
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", pad=(0.04 + 0.09) * 72 + 2, length=0)
    ax.set_yticks(np.arange(n) + 0.5, row_labels, fontsize=7)
    ax.tick_params(axis="y", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    if colorbar:
        pos = ax.get_position()
        cb = fig.colorbar(ScalarMappable(Normalize(vmin, vmax), colormaps[cmap]),
                          cax=fig.add_axes([RIGHT + 0.01, pos.y0, 0.012, max(pos.height, 0.03)]))
        cb.set_ticks(list(ticks))
        cb.ax.tick_params(labelsize=7, length=2)


def _correlation_axes(fig, ax, ex, R, classes, concepts, n_cols, colorbar):
    """Classes x concepts correlation."""
    _concept_matrix(fig, ax, ex, R.T, [_class_label(ex.class_names, c) for c in classes], concepts, n_cols, colorbar)


def concept_saliency(labels, saliency, K):
    """Mean saliency of every concept over its pixels / steps / tokens (NaN when absent)."""
    lab, sal = np.asarray(labels).ravel(), np.asarray(saliency, np.float64).ravel()
    n = np.bincount(lab, minlength=K)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, np.bincount(lab, weights=sal, minlength=K) / n, np.nan)


def _mean_saliency_axes(fig, ax, ex, mean_sal, concepts, n_cols, colorbar, rng=(-1.0, 1.0)):
    """One row: the mean saliency of every concept in this image / series / document."""
    _concept_matrix(fig, ax, ex, mean_sal[None], ["mean\nsaliency"], concepts, n_cols, colorbar, cmap="seismic",
                    vmin=rng[0], vmax=rng[1], ticks=_ticks(*rng))


def concept_row(fig, ax, ex, R, classes, mean_sal, concepts, n_cols, first, why, rng=(-1.0, 1.0)):
    """The row under a summary's explanation: correlation if available, else the concepts' mean
    saliency in this item, else a note."""
    if R is not None:
        _correlation_axes(fig, ax, ex, R, classes, concepts, n_cols, colorbar=first)
    elif mean_sal is not None:
        _mean_saliency_axes(fig, ax, ex, mean_sal, concepts, n_cols, colorbar=first, rng=rng)
    else:
        _note(ax, why if first else "")


def _note(ax, text):
    ax.axis("off")
    ax.text(0, 0.5, text, va="center", fontsize=8, color="0.35", style="italic")


NO_CLASSES = "no correlation (it needs several items and classes) and no saliency: pass classifier= to see the concepts' mean saliency"


# -- series ------------------------------------------------------------------------------------
def series_summary(ex, x, exp, sal, span, title, mean_sal=None, rng=(-1.0, 1.0)):
    import matplotlib.pyplot as plt
    a, b = (0, x.shape[1]) if span is None else (int(span[0]), int(span[1]))
    sig, labels = x[:, a:b], exp.explanation[a:b]
    unc = None if exp.uncertainty is None else exp.uncertainty[a:b]
    C, L = sig.shape
    K, pal = len(ex.palette), np.asarray(ex.palette) / 255
    names = ex.channel_names or [f"ch {c}" for c in range(C)]
    R, classes = _correlation_view(ex)
    nC = 0 if R is None else len(classes)
    n_cols = columns(K)
    blocks = [np.arange(K)[i:i + n_cols] for i in range(0, K, n_cols)]
    band_h = 0.38 * C + 0.5                                                # snippets and profile: room per channel
    heights = [0.6] * C + ([0.35] if unc is not None else []) + ([0.35] if sal is not None else [])
    heights += [0.5, max(0.5, 0.24 * nC) if nC else (0.45 if mean_sal is not None else 0.25)]   # one row, all concepts
    for _ in blocks:
        heights += [0.3, 0.85, band_h, band_h]
    fig = plt.figure(figsize=(FIG_WIDTH, sum(heights) + 1.2))
    H = fig.get_figheight()
    g = fig.add_gridspec(len(heights), 1, height_ratios=heights, hspace=0.3, left=LEFT, right=RIGHT,
                         top=1 - 0.55 / H, bottom=0.3 / H)
    if title:
        fig.text(LEFT, 1 - 0.25 / H, title, fontsize=10, fontweight="bold", va="center")

    colours = pal[labels]                                                 # the segmentation, fully opaque
    for c in range(C):
        ax = fig.add_subplot(g[c])
        lo, hi = float(sig[c].min()), float(sig[c].max())
        lo, hi = lo - 0.05 * (hi - lo + 1e-9), hi + 0.05 * (hi - lo + 1e-9)
        ax.imshow(colours[None], aspect="auto", extent=(a, b, lo, hi), interpolation="nearest")
        ax.plot(np.arange(a, b) + 0.5, sig[c], color="black", lw=0.9)
        ax.set_xlim(a, b)
        ax.set_ylim(lo, hi)
        ax.set_yticks([])
        ax.set_ylabel(names[c], rotation=0, ha="right", va="center", fontsize=8)
        last = c == C - 1 and sal is None and unc is None
        ax.tick_params(labelbottom=last, labelsize=7, length=2 if last else 0)
    row = C
    if unc is not None:                                                   # uncertainty: its own row, bright = certain
        ax = fig.add_subplot(g[row])
        ax.imshow(np.clip(unc, 0, 1)[None], aspect="auto", cmap="inferno_r", vmin=0, vmax=1, extent=(a, b, 0, 1),
                  interpolation="nearest")
        ax.set_yticks([])
        ax.set_ylabel("uncertainty", rotation=0, ha="right", va="center", fontsize=8)
        ax.tick_params(labelbottom=sal is None, labelsize=7, length=2 if sal is None else 0)
        row += 1
    if sal is not None:                                                   # the time axis goes under the last row
        ax = fig.add_subplot(g[row])
        im = ax.imshow(sal[a:b][None], aspect="auto", cmap="seismic", vmin=rng[0], vmax=rng[1], extent=(a, b, 0, 1))
        pos = ax.get_position()
        cb = fig.colorbar(im, cax=fig.add_axes([RIGHT + 0.01, pos.y0, 0.012, pos.height]))
        cb.set_ticks(_ticks(*rng))
        cb.ax.tick_params(labelsize=6, length=2)
        ax.set_yticks([])
        ax.set_ylabel("saliency", rotation=0, ha="right", va="center", fontsize=8)
        ax.tick_params(labelsize=7, length=2)
        row += 1

    row += 1                                                              # a gap
    ax = fig.add_subplot(g[row])
    concept_row(fig, ax, ex, R, classes, mean_sal, np.arange(K), K, True, NO_CLASSES, rng)
    row += 1
    for bi, concepts in enumerate(blocks):
        row += 1                                                          # a gap
        sub = g[row:row + 3].subgridspec(3, n_cols, wspace=0.1, hspace=0.12, height_ratios=[0.85, band_h, band_h])
        for j, k in enumerate(concepts):
            col = pal[k]
            axes = [fig.add_subplot(sub[r, j]) for r in range(3)]
            for ax in axes:
                _frame(ax, col)
            if j == 0:
                axes[0].set_ylabel("position\n& duration", fontsize=7)
                for ax, n in zip(axes[1:], ["snippets", "profile"]):
                    ax.set_ylabel(n, fontsize=7)
            hist = ex.positions[k]
            axes[0].fill_between(np.linspace(0, 1, len(hist) + 1)[:-1], hist, step="post", color=col, alpha=0.85)
            axes[0].set_xlim(0, 1)
            axes[0].set_ylim(0, max(hist.max(), 1) * 1.6)
            axes[0].text(0.03, 0.95, f"{k}  dur {ex.durations[k][1]:.1f}", transform=axes[0].transAxes,
                         fontsize=7, va="top")
            snips = ex.snippets[k]
            for ch in range(C):
                if snips:
                    vals = np.concatenate([v[ch] for _, _, _, v, _ in snips])
                    lo, hi = vals.min(), vals.max()
                    for _, s0, s1, v, off in snips:
                        t = np.arange(v.shape[1]) - off
                        axes[1].plot(t, (C - 1 - ch) + 0.08 + 0.84 * (v[ch] - lo) / max(hi - lo, 1e-9),
                                     color="black", lw=0.5, alpha=0.7)
                prof = ex.profiles[k, ch]
                if not np.isnan(prof).all():
                    lo, hi = np.nanmin(prof[:, 0]), np.nanmax(prof[:, 2])
                    yb = (lambda v, lo=lo, hi=hi, ch=ch: (C - 1 - ch) + 0.08 + 0.84 * (v - lo) / max(hi - lo, 1e-9))
                    axes[2].fill_between(np.arange(len(prof)), yb(prof[:, 0]), yb(prof[:, 2]), color=col, alpha=0.5, lw=0)
                    axes[2].plot(yb(prof[:, 1]), color="black", lw=0.7)
            if snips:
                axes[1].axvspan(0, np.median([s1 - s0 for _, s0, s1, _, _ in snips]), color=col, alpha=0.3, zorder=0)
            for ax in axes[1:]:
                ax.set_ylim(0, C)
        row += 3
    return fig


# -- text ----------------------------------------------------------------------------------------
TABLE_PT, DOC_PT, DOC_MIN, DOC_MAX = 7.5, 10.0, 7.0, 14.0
CHAR_IN = 0.6 * TABLE_PT / 72          # one x unit: a character of the table font, in inches
LINE_IN = 0.17                         # one y unit: a table line, in inches
MONO = "DejaVu Sans Mono"


def _span(ax, x, y, tokens, a, b, colour, context=4, max_chars=54):
    from matplotlib.patches import Rectangle
    lo, hi = max(0, a - context), min(len(tokens), b + context)
    pieces = (["…"] if lo > 0 else []) + [(i, tokens[i]) for i in range(lo, hi)] + (["…"] if hi < len(tokens) else [])
    used = 0
    for p in pieces:
        t = p if isinstance(p, str) else p[1]
        if used + len(t) > max_chars:
            ax.text(x + used, y, "…", va="center", family=MONO, fontsize=TABLE_PT)
            break
        if not isinstance(p, str) and a <= p[0] < b:
            ax.add_patch(Rectangle((x + used - 0.15, y - 0.42), len(t) + 0.3, 0.84, color=colour, lw=0))
        ax.text(x + used, y, t, va="center", family=MONO, fontsize=TABLE_PT)
        used += len(t) + 1


def text_summary(ex, tokens, exp, sal, title, mean_sal=None, rng=(-1.0, 1.0)):
    """Document (font size = |saliency|, black/white = its sign, background opacity = certainty),
    then the correlation matrix, then one row per concept."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    labels, unc = exp.explanation, exp.uncertainty
    K, pal = len(ex.palette), np.asarray(ex.palette) / 255
    R, classes = _correlation_view(ex)
    nC = 0 if R is None else len(classes)
    width_in = FIG_WIDTH * (RIGHT - LEFT)
    W = width_in / CHAR_IN                                               # x units across the axes

    sizes = np.full(len(tokens), DOC_PT)
    colours = ["black"] * len(tokens)
    if sal is not None:
        mag = np.clip(np.abs(sal) / max(abs(rng[0]), abs(rng[1])), 0, 1)  # on the chosen range's scale
        sizes = DOC_MIN + (DOC_MAX - DOC_MIN) * mag
        colours = ["black" if v >= 0 else "white" for v in sal]
    alpha = 1.0 - np.clip(unc, 0, 1) if unc is not None else np.ones(len(tokens))
    line_h = sizes.max() * 1.55 / 72 / LINE_IN                            # document line height, in table lines
    lines, line, xpos = [], [], 0.0
    for i, t in enumerate(tokens):                                       # wrap by the tokens' own widths
        w = len(t) * sizes[i] / TABLE_PT
        if xpos + w > W and line:
            lines.append(line)
            line, xpos = [], 0.0
        line.append((i, xpos, t, w))
        xpos += w + sizes[i] / TABLE_PT
    if line:
        lines.append(line)

    doc_h = (3.4 + len(lines) * line_h) * LINE_IN                      # room below for the matrix's concept numbers
    mat_h = max(0.5, 0.22 * nC) + 0.45 if nC else (0.85 if mean_sal is not None else 0.3)
    table_h = (2 + K * 3) * LINE_IN
    H = doc_h + mat_h + table_h + 1.0
    fig = plt.figure(figsize=(FIG_WIDTH, H))
    g = fig.add_gridspec(3, 1, height_ratios=[doc_h, mat_h, table_h], hspace=0.25 / (H / 3), left=LEFT, right=RIGHT,
                         top=1 - 0.2 / H, bottom=0.2 / H)

    ax = fig.add_subplot(g[0])
    n_y = 3.4 + len(lines) * line_h
    ax.set_xlim(0, W)
    ax.set_ylim(n_y, 0)
    ax.axis("off")
    ax.text(0, 0.6, title or "Document", va="center", fontsize=11, fontweight="bold")
    notes = ["background opacity = certainty"] if unc is not None else []
    if sal is not None:
        notes.append("size = |saliency|, black / white = positive / negative")
    ax.text(W, 0.6, ",  ".join(notes), ha="right", va="center", fontsize=7, color="0.35")
    for r, ln in enumerate(lines):
        y = 1.8 + (r + 0.5) * line_h
        for i, x0, t, w in ln:
            c = np.clip(pal[labels[i]] * alpha[i] + (1 - alpha[i]), 0, 1)
            ax.add_patch(Rectangle((x0 - 0.15, y - 0.45 * line_h), w + 0.3, 0.9 * line_h, color=c, lw=0))
            ax.text(x0, y, t, va="center", family=MONO, fontsize=sizes[i], color=colours[i])

    ax = fig.add_subplot(g[1])
    concept_row(fig, ax, ex, R, classes, mean_sal, np.arange(K), K, True, NO_CLASSES, rng)

    ax = fig.add_subplot(g[2])
    n_y = 2 + K * 3
    ax.set_xlim(0, W)
    ax.set_ylim(n_y, 0)
    ax.axis("off")
    x_stats, x_text = 5, 21
    ax.text(x_stats, 0.6, "spans  mean run", va="center", fontsize=6.5, family=MONO)
    ax.text(x_text, 0.6, "characteristic words  /  closest spans in context", va="center", fontsize=6.5, family=MONO)
    half = int((W - x_text) // 2 - 2)
    for k in range(K):
        y = 2 + k * 3
        ax.add_patch(Rectangle((0, y - 0.45), 2.4, 2.75, color=pal[k], lw=0))
        ax.text(2.9, y, str(k), va="center", family=MONO, fontsize=TABLE_PT)
        ax.text(x_stats, y, f"{ex.span_counts[k]:5d}  {ex.mean_runs[k]:3.1f}", va="center", family=MONO,
                fontsize=TABLE_PT)
        ax.text(x_text, y, "  ".join(ex.words[k]), va="center", fontweight="bold", family=MONO, fontsize=TABLE_PT)
        for e, (toks, a, b) in enumerate(ex.spans[k][:2]):
            _span(ax, x_text + e * (half + 2), y + 1.05, toks, a, b, pal[k], max_chars=half)
        ax.plot([0, W], [y + 2.38] * 2, color="0.9", lw=0.5)
    return fig
