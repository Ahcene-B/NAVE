# Getting started

## Install

```bash
pip install nave-xai               # numpy, onnxruntime, matplotlib; ViT-Ti/16 weights included
pip install "nave-xai[torch]"      # also your own PyTorch models, GPU
```

## 1. Explain an image

```python
import matplotlib.pyplot as plt
from matplotlib.image import imread
from nave_xai import NAVEImage

image = imread("photo.jpg")              # (H, W, 3), uint8 or float in [0, 1]; PIL images work too
explainer = NAVEImage()                       # ViT-Ti/16 trained on ImageNet, shipped with the package
exp = explainer.explain(image)

plt.imshow(explainer.palette[exp.explanation])   # every concept in its colour
```

`exp` is an `ImageExplanation`:

| | |
|---|---|
| `explanation` | `(H, W)` int32: the concept of every pixel |
| `uncertainty` | `(H, W)` float32 in [0, 1], with `explain(image, uncertainty=True)` |
| `K` | the number of concepts |
| `segmentation()` | `(H, W)` connected regions of the explanation map, like `skimage.segmentation.slic` |
| `border(width=1)` | `(H, W)` 0/1 mask of every region's inner margin |

The share of each concept in the image is `np.bincount(exp.explanation.ravel(), minlength=exp.K)`
divided by the number of pixels.

Without a previous `fit`, `explain` fits on the image it is given, at every call. After
`fit(images)`, it explains with the fitted concepts.

## 2. Draw it

nave-xai returns arrays; draw them with any tool:

```python
colours = explainer.palette[exp.explanation]                   # (H, W, 3) uint8
picture = colours.copy()
picture[exp.border(2) == 1] = 255                              # white region borders
plt.imshow(picture)

u = explainer.explain(image, uncertainty=True).uncertainty
plt.imshow((1 - u)[..., None] * colours / 255 + u[..., None])   # fades to white where uncertain
```

For a ready-made figure, `explainer.summarize(image)` shows the image, its explanation, the
correlation of the concepts with the classes, and every concept's patch.

## 3. What to explain

```python
NAVEImage(
    model="resnet50",        # "vitt16" (default), "resnet18", "resnet50", "vits16", or your own model
    training="imagenet",     # "random", "imagenet", "dino"
    layers=[3, 4],           # or "N34"; ViT: e.g. [9, 12] or "N:9-12"
    k=10,                    # concepts
    upsample="cubic",        # "nearest" (blocky, as in the paper), "linear", "cubic"
)
```

- **Layers**: shallow layers carry detail and edges, deep layers semantics; their combination is
  what makes NAVE's concepts informative. By default: N34 for ResNets, the last block for ViTs.
- **k** sets the granularity of the explanation.
- **upsample** controls how concept scores go from the feature grid to pixels.

## 4. Concepts across images

```python
explainer = NAVEImage(model="resnet50", k=10).fit(images)
explanations = explainer.predict(other_images)          # a list of Explanations
explainer.summarize(other_images[0], saliency=True)
```

See [Fitting on a dataset](datasets.md).

## 5. Preprocessing

By default, images are normalised as the model expects (ImageNet mean and standard deviation).
`transforms=` replaces that with a function, or a list of functions, from image to image:

```python
from PIL import Image
half = lambda im: np.asarray(Image.fromarray(im).resize((im.shape[1] // 2, im.shape[0] // 2)))
NAVEImage(transforms=half)        # explained at half size; maps come back at the original size
```

If the last function returns uint8 (or floats in [0, 1]), the model's normalisation follows; if it
returns normalised floats, they are used as they are. Use only resizing and colour operations.

## 6. Large images

An image goes through the encoder whole unless its longer side reaches three times the model's
input size (672 px for the built-in models). Larger images are processed in overlapping windows,
so memory stays bounded and the encoder sees them at a scale close to its training. See
[Memory and large images](memory.md).
