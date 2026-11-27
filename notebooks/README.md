# Notebooks

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ahcene-B/NAVE/blob/main/notebooks/demo.ipynb)

Every notebook runs on Google Colab: open `https://colab.research.google.com/github/Ahcene-B/NAVE/blob/main/notebooks/<path>`
(for instance `animations/01_encoder_blocks.ipynb`). Its first cell installs nave-xai (from PyPI) and what the notebook needs,
on Colab only; the training notebooks are best run with a GPU (Runtime › Change runtime type).

- `demo.ipynb`: a tour, mostly images, with short series and text sections.
- `series.ipynb`: time series (BasicMotions, a 1D ResNet trained with PyTorch).
- `text.ipynb`: text (Reuters; spaCy's encoder and a topic classifier trained with PyTorch).

- [`demo.ipynb`](demo.ipynb): a tour of nave-xai, from one explanation to datasets, saliency and your own encoder.

## Inspecting models

NAVE as a diagnostic tool, following Sections 4 and 5 of the paper:

| Notebook | Question |
|---|---|
| [`01_training`](inspecting/01_training.ipynb) | How do random, supervised and self-supervised training shape the concepts? |
| [`02_architectures`](inspecting/02_architectures.ipynb) | Do a ResNet and a ViT decompose the same image the same way? |
| [`03_depth`](inspecting/03_depth.ipynb) | How do concepts change with depth, and where is information compressed? |
| [`04_vit_registers`](inspecting/04_vit_registers.ipynb) | Do ViT high-norm artifacts form concepts, and do registers remove them? (PyTorch) |
| [`05_shortcuts`](inspecting/05_shortcuts.ipynb) | Does a shortcut take over the explanation? |
| [`06_concept_features`](inspecting/06_concept_features.ipynb) | Are concept proportions enough for a classifier probe? |
| [`07_localization`](inspecting/07_localization.ipynb) | Can a concept's region localize an object? |
| [`08_uncertainty`](inspecting/08_uncertainty.ipynb) | Where does the encoder hesitate between concepts? |
| [`09_causality`](inspecting/09_causality.ipynb) | Do the concepts that come with a class actually drive its score? Interventions in a ResNet50 and a ViT-S/16 (PyTorch) |

The first run downloads the pretrained weights. Every notebook has a `TRAINING` (or `DINO`)
setting at the top; set it to `"random"` to try it without downloading.

## Animations (`animations/`)

Animated demos (`pip install "nave-xai[animation]"`). Every sequence starts at the input of the first block
(`b0`); UMAPs keep each frame close to the previous one (`AlignedUMAP`); all panels are always shown; the
explanations of several images are computed once, so `IMAGE` only changes the drawing.

Runs on a CPU, no training:

- `01_encoder_blocks.ipynb`: inside an encoder (ViT or ResNet), block by block, block outputs and keys, the
  artifact tokens of ViTs.
- `03_augmentation.ipynb`: does the explanation survive rotation, zoom, colour shifts, noise, blur?
- `04_supervised_vs_selfsupervised.ipynb`: ImageNet and DINO training, for ViT-S/16 and ResNet50, side by side
  (downloads the weights, about 90 MB each).
- `05_image_atlas.ipynb`: UMAP of the tokens as a scatter (the chosen photo in colour) and as an atlas of patches.
- `06_artifacts.ipynb`: the artifact tokens of ViTs: norm, uncertainty, relation to the other concepts.

With PyTorch training on a GPU (each downloads its dataset once):

- `02_resnet34_training.ipynb`: a ResNet34 learning Imagenette, checkpoint by checkpoint.
- `07_shortcut.ipynb`: two classes, a mark at random places on one; a model trained with the mark and one without.
- `08_finetuning.ipynb`: an ImageNet ResNet34 fine-tuned on EuroSAT, with CKA per block from the stem.
- `09_overfitting.ipynb`: few images of two classes; a training image and a validation image side by side.
- `10_imbalance.ipynb`: one class ten times rarer: how many concepts each class gets.
