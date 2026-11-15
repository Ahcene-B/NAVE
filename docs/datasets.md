# Fitting on a dataset

`fit` extracts one set of concepts from many images; `predict` then explains any image with
them, so a concept means the same thing in every image.

```python
from nave_xai import NAVEImage

explainer = NAVEImage(model="resnet50", k=10).fit(images)     # (N, H, W, 3) array or list of images
explanations = explainer.predict(test_images)             # a list of Explanations
exp = explainer.explain(test_images[0])                   # one image: one Explanation
```

Images keep their size (larger ones are processed in windows). To resize them, use `transforms=`.

## What `fit` computes

| Attribute | |
|---|---|
| `prototypes` | `(K, D)` concept prototypes |
| `palette` | `(K, 3)` concept colours |
| `patches` | `(K, 2, 32, 32, 3)` closest and largest cell of every concept |
| `classes` | the groups given to `fit`, else the model's predicted classes |
| `correlation` | `(K, C)` correlation of concept proportions with `classes` |
| `fitted` | `True` |

With `fit_on="all"` (default) every composite pixel is used: exactly when they fit in half the
free memory, otherwise streamed through mini-batch k-means, with a warning (PCA is accumulated
exactly either way). `fit_on="sample"` uses `sample_size` random composite pixels.

Classes come from the model's own head (the built-in ImageNet models, PyTorch models that output
class scores) or from `classifier=`: a `torch.nn.Linear`, a weight matrix, or a function from image
to class scores.

## Groups

```python
explainer.fit(images, groups=labels)                       # one projector per group
explainer.fit(images, groups=labels, merge_groups=False)   # keep them apart
explainer.predict(images, groups=labels)                   # each image with its group's concepts
```

This is the paper's protocol: one k-means per class, merged. Concepts are numbered by group, and
every group gets a colour family; `classes` and `correlation` use the groups. With
`merge_groups=False`, `predict(groups=...)` uses only each image's group; without groups every
projector scores every pixel and the closest concept wins.

## Ordering and colours

```python
explainer.reorder("ward")                                       # or "average", "single", or a list
explainer.reorder(by="classes", recolor=True)                   # grouped and coloured by class
explainer.reorder(by="classes", recolor=True, color_vary=["lightness", "saturation"], color_spread=0.7)
explainer.set_palette("tab20")                                  # Spectral by default; or a (K, 3) uint8 array
```

Without `recolor`, colours follow their concepts.

## Patches

```python
explainer.update_patches(other_images, min_size=48)      # cells at least 48 x 48 px for "closest"
```

If no cell is large enough, the largest is used (with a warning); a concept without any cell gets
an empty patch.

## Saliency and summary

```python
maps = explainer.saliency(images)                   # one (H, W) map per image
explainer.summarize(image, uncertainty=True, saliency=True, patches="closest")
```

`summarize` draws, for one image, the image | its explanation | its saliency, all at the image's
size; then, for every block of 10 concepts, their correlation with the classes and their patches
framed in their colour, as in the paper's Figure 1.

The correlation needs several images with at least two different classes. When there is none (one
image, or a single class), that row shows instead each concept's **mean saliency** in the summarized
image (with a classifier), so the concepts can still be related to the prediction. Saliency is a
cosine and is always shown on [-1, 1]. With `uncertainty=True`, uncertain pixels of
the explanation fade to white.
