# NAVE Superpixels for GIMP 3

Superpixels from a vision encoder's internal features, based on NAVE
(Neuro-Activated Vision Explanations) by Ahcène Boubekki, Samuel G. Fadel and Sebastian Mair.

The plugin clusters composite feature activations from several encoder depths with k-means,
following Algorithm 1 of the paper, and turns the clusters into superpixels inside GIMP.
It is an independent design tool, not the authors' reference implementation: images are
processed in overlapping windows, which the paper does not do, so results can differ from
the paper's.

## Install
Copy the `nave` folder (this folder's `nave/`) into GIMP's per-user plug-ins folder and restart GIMP:

- Linux: `~/.config/GIMP/3.0/plug-ins/nave/`
- macOS: `~/Library/Application Support/GIMP/3.0/plug-ins/nave/`
- Windows: `%APPDATA%\GIMP\3.0\plug-ins\nave\`

On Linux and macOS: `chmod +x nave.py`.

The plugin is under **Filters > Generic > NAVE Superpixels...**

Models: ResNet18 (random or ImageNet), ResNet50 and ViT-S/16 (random, ImageNet or DINO).
Tick the blocks to combine; the checkboxes shown follow the chosen model.

- **Window size** (224-2048 px, multiples of 32): the image is processed in overlapping windows.
- **Two passes**: one clustering for the whole image instead of one per window. Runs the
  network twice; needed for the segmentation layer.
- **Outputs** (any combination, one layer each): borders, colored superpixels (each segment
  in its average color), segmentation (each NAVE cluster in its own color).

Before running, the plugin estimates the memory needed and warns you if it may not fit.
Memory grows with the window size, very quickly for ViT, and with block 1 for ResNet.

## First run
The plugin needs a Python 3.10+ with numpy and onnxruntime. It looks for a local
Python and builds a private environment from it. If none is usable, it offers to
download one with uv (about 100 MB). Everything goes in one folder; delete it to
uninstall:

- Linux: `~/.cache/nave`
- macOS: `~/Library/Caches/nave`
- Windows: `%LOCALAPPDATA%\nave`

Pretrained weights are downloaded there the first time each model/training is used
(about 90-100 MB each). Random weights need no download.

Weights you already have are reused instead of downloaded:

- Hugging Face cache (timm): `~/.cache/huggingface/hub`, or `HF_HUB_CACHE` / `HF_HOME`
- torch hub cache (torchvision, DINO, DeiT): `~/.cache/torch/hub/checkpoints`, or `TORCH_HOME`

## Using your own Python (e.g. conda)
Put the full path of a Python that has numpy and onnxruntime in **External Python**,
for example `~/miniconda3/envs/nave/bin/python`.

## Files
- `nave.py`: GIMP plugin (standard library only)
- `nave_env.py`: Python environment setup
- `nave_mem.py`: memory estimate for the warning
- `nave_core.py`: the worker: windows, NAVE, k-means, segments, rendering
- `nave_models.py`: weights (safetensors, .pth without torch, random) and onnxruntime
- `graphs/`: weight-less ONNX graphs of ResNet18, ResNet50 and ViT-S/16

## Credits
This plugin implements the method of:

> Ahcène Boubekki, Samuel G. Fadel, Sebastian Mair. *Multi-depth Concept Extraction for Post-Hoc Vision
> Encoder Explanation.* Machine Learning and Knowledge Discovery in Databases (ECML PKDD), Research Track,
> Springer, 2026, pp. 449–467. https://doi.org/10.1007/978-3-032-37664-0_26
> Code: https://github.com/Ahcene-B/NAVE

If you use this plugin in research, please cite the paper:

```bibtex
@inproceedings{boubekki2026nave,
  author    = {Boubekki, Ahc{\`e}ne and Fadel, Samuel G. and Mair, Sebastian},
  editor    = {Ceci, Michelangelo and Kocev, Dragi and Monreale, Anna and Vens, Celine and Flach, Peter
               and Manco, Giuseppe and Spiliopoulou, Myra and Masciari, Elio},
  title     = {Multi-depth Concept Extraction for Post-Hoc Vision Encoder Explanation},
  booktitle = {Machine Learning and Knowledge Discovery in Databases. Research Track},
  publisher = {Springer Nature Switzerland},
  address   = {Cham},
  pages     = {449--467},
  year      = {2026},
  doi       = {10.1007/978-3-032-37664-0_26},
  isbn      = {978-3-032-37664-0}
}
```

Pretrained weights are downloaded from their original sources and keep their own licenses:

- ResNet18 / ResNet50 ImageNet weights: torchvision (BSD-3-Clause), via timm on Hugging Face
- DINO ResNet50 and ViT-S/16: Caron et al., *Emerging Properties in Self-Supervised Vision
  Transformers* (Apache-2.0)
- DeiT-S (ImageNet ViT-S/16): Touvron et al., *Training data-efficient image transformers*
  (Apache-2.0), via timm

Runtime dependencies: numpy (BSD-3-Clause), onnxruntime (MIT), and optionally uv
(MIT / Apache-2.0) to set up Python.

## License
MIT, see `LICENSE`. This covers the plugin's code only, not the downloaded weights.

The graphs in `nave/graphs/` are the nave-xai package's (`tools/build_graphs.py` at the root of this
repository rebuilds them, with `pip install onnx`); the plugin uses their block outputs `b1`, `b2`, ...

This plugin is part of the [nave-xai](https://github.com/Ahcene-B/NAVE) repository, under the same MIT license.
