# SPDX-License-Identifier: MIT
"""The figure of `NAVEImage.summarize`, in the style of the paper's Figure 1.

Row 1: the image | its explanation (uncertainty as opacity on white) | its saliency, all the same size.
Then, per block of up to 10 concepts: their correlation with the classes (rows), and below it their
patches, framed in the concept's colour, one per correlation column.
"""
import numpy as np

from ._layout import FIG_WIDTH, LEFT, RIGHT, columns

CLASS_LIMIT = 30


def _clean(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)


def _explanation_picture(exp, palette):
    colours = np.asarray(palette)[exp.explanation]
    if exp.uncertainty is None:
        return colours
    a = (1.0 - np.clip(exp.uncertainty, 0, 1))[..., None]          # opacity = certainty, on white
    return np.round(colours * a + 255.0 * (1 - a)).astype(np.uint8)


def _patch_strip(patches, palette, kind, concepts, n_cols, pad=3, gap=4):
    """The patches of `concepts` side by side, each framed in its colour, n_cols columns wide."""
    side = patches.shape[2]
    cell = side + 2 * pad
    out = np.full((cell, n_cols * (cell + gap), 3), 255, np.uint8)  # one (cell + gap) column per concept
    for j, k in enumerate(concepts):
        x = j * (cell + gap) + gap // 2
        out[:, x:x + cell] = np.asarray(palette)[k]
        out[pad:pad + side, x + pad:x + pad + side] = patches[k, kind]
    return out


def _spines_off(ax):
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)


def summarize(explainer, rgb, exp, saliency, show_correlation, kind, title="", mean_sal=None):
    import matplotlib.pyplot as plt
    from .interp import resample

    palette, K = explainer.palette, len(explainer.palette)
    R, classes = explainer.correlation, explainer.classes
    show_correlation = show_correlation and R is not None
    if show_correlation and len(classes) > CLASS_LIMIT:
        R, classes = R[:, :CLASS_LIMIT], classes[:CLASS_LIMIT]
    C = 0 if not show_correlation else len(classes)
    n_cols = columns(K)
    blocks = [np.arange(K)[i:i + n_cols] for i in range(0, K, n_cols)]

    width = FIG_WIDTH
    column = (RIGHT - LEFT) * width / n_cols                          # inches per concept
    heights = [3.4]
    show_row = show_correlation or mean_sal is not None
    for _ in blocks:
        heights += ([max(0.6, 0.24 * C) + 0.25 if show_correlation else 0.7] if show_row else []) + [min(column, 1.4)]
    fig = plt.figure(figsize=(width, sum(heights) + 0.6 * len(heights) + (0.35 if title else 0)))
    top_in = 0.75 if title else 0.4                                    # room for the title above the panels' own
    grid = fig.add_gridspec(len(heights), 1, height_ratios=heights, hspace=0.45, left=LEFT, right=RIGHT,
                            top=1 - top_in / fig.get_figheight(), bottom=0.5 / fig.get_figheight())

    if title:
        fig.text(LEFT, 1 - 0.2 / fig.get_figheight(), title, fontsize=10, fontweight="bold", va="center")

    # row 1: image | explanation | saliency, all at the image's size
    n = 3 if saliency is not None else 2
    top = grid[0].subgridspec(1, n, wspace=0.12)
    ax = fig.add_subplot(top[0])
    ax.imshow(rgb)
    ax.set_title("image", fontsize=10)
    _clean(ax)
    ax = fig.add_subplot(top[1])
    ax.imshow(_explanation_picture(exp, palette))
    ax.set_title("NAVE explanation" + (" (opacity = certainty)" if exp.uncertainty is not None else ""),
                 fontsize=10)
    _clean(ax)
    if saliency is not None:
        if saliency.shape != rgb.shape[:2]:
            saliency = resample(saliency, rgb.shape[0], rgb.shape[1], "linear")
        ax = fig.add_subplot(top[2])
        im = ax.imshow(saliency, cmap="seismic", vmin=-1, vmax=1)          # a cosine: always on [-1, 1]
        ax.set_title("saliency", fontsize=10)
        _clean(ax)
        pos = ax.get_position()                                             # outside the right margin
        bar = fig.colorbar(im, cax=fig.add_axes([RIGHT + 0.01, pos.y0, 0.012, pos.height]))
        bar.set_ticks([-1, 0, 1])
        bar.ax.tick_params(labelsize=7)

    # per block of concepts: correlation (classes x concepts, each column headed by the concept's
    # colour and number, as in the series and text summaries), then the patches
    from ._figure_sequence import concept_row
    row = 1
    for b, concepts in enumerate(blocks):
        if show_row:
            ax = fig.add_subplot(grid[row])
            concept_row(fig, ax, explainer, R if show_correlation else None, classes, mean_sal, concepts, n_cols,
                        b == 0, "")
            if show_correlation:
                ax.set_ylabel("class", fontsize=9)
            row += 1
        ax = fig.add_subplot(grid[row])
        ax.imshow(_patch_strip(explainer.patches, palette, kind, concepts, n_cols), aspect="auto",
                  interpolation="nearest", extent=(0, n_cols, 1, 0))
        ax.set_xticks(np.arange(len(concepts)) + 0.5, [str(k) for k in concepts], fontsize=8)
        ax.set_yticks([])
        ax.tick_params(length=0)
        _spines_off(ax)
        if b == len(blocks) - 1:
            ax.set_xlabel(f"concepts ({'closest' if kind == 0 else 'largest'} cell)", fontsize=9)
        row += 1
    return fig


def _has(names, c):
    try:
        names[c]
        return True
    except (KeyError, IndexError, TypeError):
        return False
