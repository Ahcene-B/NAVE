# Notebooks

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

The first run downloads the pretrained weights. Every notebook has a `TRAINING` (or `DINO`)
setting at the top; set it to `"random"` to try it without downloading.
