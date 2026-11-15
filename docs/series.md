# Time series

`NAVESeries` explains encoders of time series (1D CNNs, transformers): one concept per time step,
shared by all channels, since the encoder mixes them. The ECG version of the method is described in
Boubekki et al., *Explaining deep learning for ECG using time-localized clusters* (arXiv:2509.15198).

```python
from nave_xai import NAVESeries

explainer = NAVESeries(model, layers=["block3", "block4", "block5"], k=20, multiple=8).fit(series)
exp = explainer.explain(series[0], uncertainty=True)
exp.explanation          # (T,) the concept of every time step
exp.segments()           # (T,) runs of steps with the same concept
exp.border()             # (T,) 1 at change points
explainer.summarize(series[0], uncertainty=True, saliency=True)
```

## Inputs and models

Series are channels first, as in PyTorch: an `(N, C, T)` array, a list of `(C, T_i)` arrays of
different lengths, or one `(C, T)` or `(T,)` series. `transforms=` applies functions to each
`(C, T)` series (normalisation, filtering); there is none by default.

`model` is a `torch.nn.Module` (layers are module names; `channels_last=True` for transformer
layers that output `(N, t, C)`), an `.onnx` file (layers are output names), or any function
`x (1, C, T) -> {layer: (C_b, t_b) array}`; a function may also return the class scores under
`"logits"`. `multiple=` is the model's total downsampling: inputs are padded (mirrored) to a
multiple of it.

## Long series

With `window=W` (steps), series longer than W are processed in windows of W steps every W/2
steps; every step comes from the window whose centre is nearest, far from any window edge.

## What `fit` computes

As for images, `prototypes`, `palette`, `classes` (groups or the model's predictions) and
`correlation`; instead of image patches, for every concept:

| | |
|---|---|
| `snippets` | the segments closest to the prototype, with context, all channels |
| `profiles` | `(K, C, 40, 3)`: 10th, 50th and 90th percentiles of its segments, resampled to 40 points |
| `positions` | `(K, 12)`: where its segments sit in the series |
| `durations` | `(K, 2)`: median and mean segment length, in steps |

`update_snippets(series)` recomputes them on other series.

## Summary

`summarize(series)` draws one row per channel with the signal over the segmentation, an uncertainty
row (`uncertainty=True`; bright = certain, black = uncertain), a saliency row (`saliency=True`, standardized then passed through a
sigmoid, as in the ECG paper), the correlation of the concepts with the classes, and for every
concept its position and duration, its snippets and its profile, one band per channel, 10 concepts
per row. `span=(start, end)` zooms on part of a long series; `label=` adds the true class next to the
model's prediction in the title.

See `notebooks/series.ipynb` for a complete example on BasicMotions (smartwatch motions).

Without correlation (one item, or a single class), the summary shows each concept's mean saliency in
the item instead, on [-1, 1] (it needs a classifier).
