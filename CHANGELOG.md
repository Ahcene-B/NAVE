# Changelog

## 0.1.2

- The GIMP 3 plugin in `gimp/`, with the package's model graphs (its ViT graph now works on
  onnxruntime 1.20 and newer).
- Animated demo notebooks (`notebooks/animations/`), shown in the notebook and saved as GIFs:
  - `01_vit_blocks.ipynb`: a ViT's patch tokens block by block, with UMAP (the last block or the
    previous one as reference), coloured by the last block's concepts, and optionally the NAVE
    explanation, uncertainty and NAVE saliency of an image at every block, from the block outputs and
    from the attention keys; a section following the artifact tokens;
  - `03` to `10`: augmentation, supervised against self-supervised, an image atlas of the tokens, the
    artifact tokens, a learned shortcut, fine-tuning with CKA, overfitting, class imbalance;
  - sliders (`ipywidgets`) in the ViT and training notebooks;
  - `02_resnet34_training.ipynb`: a ResNet34 trained from scratch on Imagenette, the same views from
    one checkpoint to the next, with the probe accuracy.
- Built-in models: `b0`, the input of block 1 (ResNets: the stem's output; ViTs: patch and position
  embeddings), usable as a layer.
- `animation` extra: `pip install "nave-xai[animation]"` (umap-learn, scikit-image).
- Built-in ViTs: `features="keys"`; summaries: `saliency_range`, `label=`, mean saliency per concept
  when there is no correlation, fixed width.

## 0.1.0

First release.

- `NAVESeries` (time series, 1D CNNs and transformers) and `NAVEText` (one concept per token; spaCy's
  encoder built in with the `text` extra), sharing the projectors, ordering and colours of images.
- Built-in ViTs: `features="keys"` reads the attention keys of the chosen blocks.
- `NAVEImage`: one object for everything. `explain` (fits afresh on the images unless `fit` was called),
  `fit`, `predict`, `fit_predict`, `segment`, `reorder`, `set_palette`, `update_patches`,
  `saliency` and `summarize` (a figure in the style of the paper's Figure 1).
- `ImageExplanation`: the explanation map (`explanation`), the optional `uncertainty`, `K`, and
  `segmentation()` and `border()`.
- Projectors: k-means (exact on all composite pixels, or streamed), Bayesian k-means (Gibbs
  sampling, Dirichlet prior), PCA, or any scikit-learn-style clusterer or projection.
- Groups: one projector per group (the paper's protocol), merged or kept apart; concepts ordered
  and coloured by group.
- Fitted attributes: `prototypes`, `palette`, `patches` (closest and largest cell per concept),
  `classes` (groups or predictions) and `correlation`.
- Ordering: Ward, average or single linkage of the prototypes, or by class; colour families per class.
- Built-in encoders without torch: ViT-Ti/16 (ImageNet weights shipped), ResNet18, ResNet50 and
  ViT-S/16 (random, ImageNet, DINO); PyTorch modules, ONNX files and functions as backbones.
- Large images in windows twice the model's input size, every half window, each pixel from the
  window whose centre is nearest; memory estimates and warnings.
