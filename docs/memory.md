# Memory and large images

## How images are processed

| Image | `window="auto"` does |
|---|---|
| longer side below `window_factor` (3) times the model's input size (672 px for the built-in models) | the whole image in one forward pass |
| larger | windows twice the input size (448 px), every half window (224 px) |

Every pixel comes from the window whose centre is nearest, so no pixel is read close to a window's
edge, where features are distorted: each window keeps its central half (margins of about a quarter
of the window, rounded to the model's feature grid). Concepts are fitted on all windows together,
so they mean the same everywhere in the image. Models tied to one input size use windows of that
size.

The network also needs image sides that are multiples of its downsampling (32 for ResNets, 16 for
ViTs), so images are padded by mirroring their edges, and the maps cropped back.

The class an image is predicted to be (for `classes` and `correlation`) is the average of the
windows' logits, each weighted by its share of real, not padded, pixels. Whole-image methods
(`saliency`) run on the image resized to at most 1024 px, and their map is resized back.

```python
NAVEImage(window=None)           # always whole
NAVEImage(window=640)            # your window size (a multiple of 32)
NAVEImage(window_factor=5)       # windows only from five times the input size
```

## Memory

Composite pixels are produced a few rows at a time (about 64 MB per chunk), so peak memory is set
by one window, not by the image. What grows:

- **ViT** attention grows with the square of the number of patches in a window.
- **ResNet block 1** has a grid 16 times larger than block 3.
- **`fit_on="all"`** stores every composite pixel when they fit in half the free memory, and
  otherwise streams them through mini-batch k-means, with a warning.

Before a run, nave-xai estimates its peak memory and issues a `NaveMemoryWarning` if it may not fit.

```python
import warnings
from nave_xai import NaveMemoryWarning
warnings.simplefilter("error", NaveMemoryWarning)      # turn the warning into an error
```
